import os
import tempfile
import multiprocessing
from datetime import date
from collections import namedtuple

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from openai import deepq
from openai.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from openai.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from openai.deepq.models import build_q_func

from deepq import StatePreprocessor, ActionAdviceRewardShaper
from dotaenv import DotaEnvironment

REPLAY_DIR = os.path.join('..', 'completed-observations')
MIN_STEPS_TO_FOLLOW_DEMO_FOR = 0


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


UpdateMessage = namedtuple('UpdateMessage', ['status', 'transition', 'demo_picked'])
UPDATE_STATUS_CONTINUE = 0
UPDATE_STATUS_SEND_WEIGHTS = 1
UPDATE_STATUS_FINISH = 2


def save_demo_switching_stats(demo_switching_stats, dir, num_episodes):
    demo_switching_filename = 'demo_switching_stats_{}'.format(num_episodes)
    demo_switching_path = os.path.join(dir, demo_switching_filename)
    with open(demo_switching_path, 'w') as foutput:
        print('\n'.join(map(str, demo_switching_stats)), file=foutput)


def do_network_training(
        updates_queue: multiprocessing.Queue,
        weights_queue: multiprocessing.Queue,
        network, seed, lr, total_timesteps, learning_starts,
        buffer_size, exploration_fraction, exploration_initial_eps, exploration_final_eps,
        train_freq, batch_size, print_freq, checkpoint_freq, gamma,
        target_network_update_freq, prioritized_replay, prioritized_replay_alpha,
        prioritized_replay_beta0, prioritized_replay_beta_iters,
        prioritized_replay_eps, experiment_name, load_path, network_kwargs):
    _ = get_session()
    set_global_seeds(seed)
    q_func = build_q_func(network, **network_kwargs)

    def make_obs_ph(name):
        return ObservationInput(DotaEnvironment.get_observation_space(), name=name)

    _, train, update_target, debug = deepq.build_train(
        scope='deepq_train',
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=DotaEnvironment.get_action_space().n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,)

    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    U.initialize()
    update_target()

    reward_shaper = ActionAdviceRewardShaper(
        replay_dir=REPLAY_DIR,
        max_timesteps=total_timesteps + 18000)  # Plus max episode length
    reward_shaper.load()
    reward_shaper.generate_merged_demo()

    full_exp_name = '{}-{}'.format(date.today().strftime('%Y%m%d'), experiment_name)
    experiment_dir = os.path.join('experiments', full_exp_name)
    os.makedirs(experiment_dir, exist_ok=True)

    learning_dir = os.path.join(experiment_dir, 'learning')
    learning_summary_writer = tf.summary.FileWriter(learning_dir)

    update_step_t = 0
    should_finish = False
    while not should_finish:
        message = updates_queue.get()
        logger.log('Got message in do_network_training')
        if message.status == UPDATE_STATUS_CONTINUE:
            transition = message.transition
            replay_buffer.add(*transition)
            next_act_step = transition[5] + 1
            reward_shaper.set_demo_picked(next_act_step, message.demo_picked)

            if update_step_t >= learning_starts and update_step_t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(update_step_t))
                    (obses_t, actions, rewards, obses_tp1, dones, ts, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones, ts = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                biases_t = []
                for obs_t, timestep in zip(obses_t, ts):
                    biases_t.append(reward_shaper.get_action_potentials(obs_t, timestep))
                biases_tp1 = []
                for obs_tp1, timestep in zip(obses_tp1, ts):
                    biases_tp1.append(reward_shaper.get_action_potentials(obs_tp1, timestep + 1))
                td_errors, weighted_error = train(
                    obses_t, biases_t, actions, rewards, obses_tp1, biases_tp1, dones, weights)
                # Loss logging
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='weighted_error', simple_value=weighted_error)])
                learning_summary_writer.add_summary(summary, update_step_t)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            if update_step_t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()
            update_step_t += 1
        elif message.status == UPDATE_STATUS_SEND_WEIGHTS:
            q_func_vars = get_session().run(debug['q_func_vars'])
            weights_queue.put(q_func_vars)
        elif message.status == UPDATE_STATUS_FINISH:
            should_finish = True
        else:
            logger.log(f'Unknown status in UpdateMessage: {message.status}')


def do_agent_exploration(
        updates_queue: multiprocessing.Queue,
        q_func_vars_trained_queue: multiprocessing.Queue,
        network, seed, lr, total_timesteps, learning_starts,
        buffer_size, exploration_fraction, exploration_initial_eps, exploration_final_eps,
        train_freq, batch_size, print_freq, checkpoint_freq, gamma,
        target_network_update_freq, prioritized_replay, prioritized_replay_alpha,
        prioritized_replay_beta0, prioritized_replay_beta_iters,
        prioritized_replay_eps, experiment_name, load_path, network_kwargs):
    env = DotaEnvironment()

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, _, _, debug = deepq.build_train(
        scope='deepq_act',
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10, )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n, }

    act = ActWrapper(act, act_params)

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=exploration_initial_eps,
                                 final_p=exploration_final_eps)

    U.initialize()

    reward_shaper = ActionAdviceRewardShaper(
        replay_dir=REPLAY_DIR,
        max_timesteps=total_timesteps + 18000)  # Plus max episode length
    reward_shaper.load()
    reward_shaper.generate_merged_demo()

    full_exp_name = '{}-{}'.format(date.today().strftime('%Y%m%d'), experiment_name)
    experiment_dir = os.path.join('experiments', full_exp_name)
    os.makedirs(experiment_dir, exist_ok=True)

    summary_dir = os.path.join(experiment_dir, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)
    summary_writer = tf.summary.FileWriter(summary_dir)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    stats_dir = os.path.join(experiment_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_dir or td

        os.makedirs(td, exist_ok=True)
        model_file = os.path.join(td, "best_model")
        model_saved = False
        saved_mean_reward = None

        # if os.path.exists(model_file):
        #     print('Model is loading')
        #     load_variables(model_file)
        #     logger.log('Loaded model from {}'.format(model_file))
        #     model_saved = True
        # elif load_path is not None:
        #     load_variables(load_path)
        #     logger.log('Loaded model from {}'.format(load_path))

        def synchronize_q_func_vars():
            updates_queue.put(UpdateMessage(UPDATE_STATUS_SEND_WEIGHTS, None, None))
            q_func_vars_trained = q_func_vars_trained_queue.get()
            update_q_func_expr = []
            for var, var_trained in zip(debug['q_func_vars'], q_func_vars_trained):
                update_q_func_expr.append(var.assign(var_trained))
            update_q_func_expr = tf.group(*update_q_func_expr)
            sess.run(update_q_func_expr)
        synchronize_q_func_vars()

        episode_rewards = []
        act_step_t = 0
        while act_step_t < total_timesteps:
            # Reset the environment
            obs = env.reset()
            obs = StatePreprocessor.process(obs)
            episode_rewards.append(0.0)
            done = False
            # Demo preservation variables
            demo_picked = 0
            demo_picked_step = 0
            # Demo switching statistics
            demo_switching_stats = [(0, 0)]
            # Sample the episode until it is completed
            act_started_step_t = act_step_t
            while not done:
                # Take action and update exploration to the newest value
                biases, demo_indexes = reward_shaper.get_action_potentials_with_indexes(obs,
                                                                                        act_step_t)
                update_eps = exploration.value(act_step_t)
                actions, is_randoms = act(np.array(obs)[None], biases, update_eps=update_eps)
                action, is_random = actions[0], is_randoms[0]
                if not is_random:
                    bias_demo = demo_indexes[action]
                    if bias_demo != demo_switching_stats[-1][1]:
                        demo_switching_stats.append((act_step_t - act_started_step_t, bias_demo))
                    if bias_demo != 0 and demo_picked == 0:
                        demo_picked = bias_demo
                        demo_picked_step = act_step_t + 1
                pairs = env.step(action)
                action, (new_obs, rew, done, _) = pairs[-1]
                logger.log(f'{act_step_t}/{total_timesteps} obs {obs} action {action}')

                # Compute state on the real reward but learn from the normalized version
                episode_rewards[-1] += rew
                rew = np.sign(rew) * np.log(1 + np.abs(rew))
                new_obs = StatePreprocessor.process(new_obs)

                if len(new_obs) == 0:
                    done = True
                else:
                    transition = (obs, action, rew, new_obs, float(done), act_step_t)
                    obs = new_obs
                    act_step_t += 1
                    if act_step_t - demo_picked_step >= MIN_STEPS_TO_FOLLOW_DEMO_FOR:
                        demo_picked = 0
                    reward_shaper.set_demo_picked(act_step_t, demo_picked)
                    updates_queue.put(
                        UpdateMessage(UPDATE_STATUS_CONTINUE, transition, demo_picked))
            # Post episode logging
            summary = tf.Summary(
                value=[tf.Summary.Value(tag="rewards", simple_value=episode_rewards[-1])])
            summary_writer.add_summary(summary, act_step_t)
            summary = tf.Summary(
                value=[tf.Summary.Value(tag="eps", simple_value=update_eps)])
            summary_writer.add_summary(summary, act_step_t)
            summary = tf.Summary(
                value=[tf.Summary.Value(tag="episode_steps",
                                        simple_value=act_step_t - act_started_step_t)])
            summary_writer.add_summary(summary, act_step_t)
            mean_5ep_reward = round(float(np.mean(episode_rewards[-5:])), 1)
            num_episodes = len(episode_rewards)
            if print_freq is not None and num_episodes % print_freq == 0:
                logger.record_tabular("steps", act_step_t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 5 episode reward", mean_5ep_reward)
                logger.record_tabular("% time spent exploring",
                                      int(100 * exploration.value(act_step_t)))
                logger.dump_tabular()
            # Wait for the learning to finish and synchronize
            synchronize_q_func_vars()
            # Record demo_switching_stats
            save_demo_switching_stats(demo_switching_stats, stats_dir, num_episodes)
            if checkpoint_freq is not None and num_episodes % checkpoint_freq == 0:
                # Periodically save the model
                rec_model_file = os.path.join(td, "model_{}_{:.2f}".format(num_episodes,
                                                                           mean_5ep_reward))
                save_variables(rec_model_file)
                # Check whether the model is the best so far
                if saved_mean_reward is None or mean_5ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward, mean_5ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_5ep_reward

        updates_queue.put(UpdateMessage(UPDATE_STATUS_FINISH, None, None))


def learn(network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          learning_starts=1000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_initial_eps=1.0,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=100,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          experiment_name='unnamed',
          load_path=None,
          **network_kwargs):
    """Train a deepq model.

    Parameters
    -------
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    experiment_name: str
        name of the experiment (default: trial)
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    multiprocessing.set_start_method('spawn')
    updates_queue = multiprocessing.Queue()
    q_func_vars_trained_queue = multiprocessing.Queue()

    exploration_process = multiprocessing.Process(
        target=do_agent_exploration,
        args=(updates_queue, q_func_vars_trained_queue,
              network, seed, lr, total_timesteps, learning_starts,
              buffer_size, exploration_fraction, exploration_initial_eps, exploration_final_eps,
              train_freq, batch_size, print_freq, checkpoint_freq, gamma,
              target_network_update_freq, prioritized_replay, prioritized_replay_alpha,
              prioritized_replay_beta0, prioritized_replay_beta_iters,
              prioritized_replay_eps, experiment_name, load_path, network_kwargs))
    exploration_process.daemon = True
    exploration_process.start()

    training_process = multiprocessing.Process(
        target=do_network_training,
        args=(updates_queue, q_func_vars_trained_queue,
              network, seed, lr, total_timesteps, learning_starts,
              buffer_size, exploration_fraction, exploration_initial_eps, exploration_final_eps,
              train_freq, batch_size, print_freq, checkpoint_freq, gamma,
              target_network_update_freq, prioritized_replay, prioritized_replay_alpha,
              prioritized_replay_beta0, prioritized_replay_beta_iters,
              prioritized_replay_eps, experiment_name, load_path, network_kwargs))
    training_process.daemon = True
    training_process.start()

    training_process.join()
    exploration_process.join()

    updates_queue.close()
    updates_queue.join_thread()
    q_func_vars_trained_queue.close()
    q_func_vars_trained_queue.join_thread()
