import os
import tempfile

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


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_initial_eps=1.0,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=1,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          experiment_name='unnamed',
          load_path=None,
          **network_kwargs
          ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
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
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
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
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
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
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
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
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=exploration_initial_eps,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    reward_shaper = ActionAdviceRewardShaper('../completed-observations')
    reward_shaper.load()

    experiment_dir = os.path.join('experiments', experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    summary_dir = os.path.join(experiment_dir, 'summaries')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir)

    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_dir or td

        if not os.path.exists(td):
            os.makedirs(td)
        model_file = os.path.join(td, "best_model")
        model_saved = False
        saved_mean_reward = None

        if os.path.exists(model_file):
            print('Model is loading')
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        episode_rewards = []
        update_step_t = 0
        while update_step_t < total_timesteps:
            # Reset the environment
            obs = env.reset()
            obs = StatePreprocessor.process(obs)
            episode_rewards.append(0.0)
            reset = True
            done = False
            # Sample the episode until it is completed
            act_step_t = update_step_t
            while not done:
                if callback is not None:
                    if callback(locals(), globals()):
                        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not param_noise:
                    update_eps = exploration.value(act_step_t)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(act_step_t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(
                        1. - exploration.value(act_step_t) +
                        exploration.value(act_step_t) / float(env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                biases = reward_shaper.get_action_potentials([obs])
                action = act(np.array(obs)[None], biases, update_eps=update_eps, **kwargs)[0]
                reset = False

                pairs = env.step(action)
                action, (new_obs, rew, done, _) = pairs[-1]
                episode_rewards[-1] += rew
                new_obs = StatePreprocessor.process(new_obs)

                logger.log('{}/{} obs {} action {}'.format(act_step_t, total_timesteps, obs, action))
                act_step_t += 1
                if len(new_obs) == 0:
                    done = True
                else:
                    replay_buffer.add(obs, action, rew, new_obs, float(done))
                    obs = new_obs
            # Post episode logging
            summary = tf.Summary(value=[tf.Summary.Value(tag="rewards", simple_value=episode_rewards[-1])])
            summary_writer.add_summary(summary, act_step_t)
            summary = tf.Summary(value=[tf.Summary.Value(tag="eps", simple_value=update_eps)])
            summary_writer.add_summary(summary, act_step_t)
            summary = tf.Summary(value=[tf.Summary.Value(tag="episode_steps", simple_value=act_step_t-update_step_t)])
            summary_writer.add_summary(summary, act_step_t)
            mean_5ep_reward = round(np.mean(episode_rewards[-5:]), 1)
            num_episodes = len(episode_rewards)
            if print_freq is not None and num_episodes % print_freq == 0:
                logger.record_tabular("steps", act_step_t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 5 episode reward", mean_5ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(act_step_t)))
                logger.dump_tabular()
            if checkpoint_freq is not None and num_episodes % checkpoint_freq == 0:
                # Periodically save the model
                rec_model_file = os.path.join(td, "model_{}_{:.2f}".format(num_episodes, mean_5ep_reward))
                save_variables(rec_model_file)
                # Check whether it is best
                if saved_mean_reward is None or mean_5ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward, mean_5ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_5ep_reward
            # Do the learning
            while update_step_t < min(act_step_t, total_timesteps):
                if update_step_t % train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if prioritized_replay:
                        experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(update_step_t))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None
                    biases_t = reward_shaper.get_action_potentials(obses_t)
                    biases_tp1 = reward_shaper.get_action_potentials(obses_tp1)
                    td_errors, weighted_error = train(
                        obses_t, biases_t, actions, rewards, obses_tp1, biases_tp1, dones, weights)

                    # Loss logging
                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag='weighted_error', simple_value=weighted_error)])
                    summary_writer.add_summary(summary, update_step_t)

                    if prioritized_replay:
                        new_priorities = np.abs(td_errors) + prioritized_replay_eps
                        replay_buffer.update_priorities(batch_idxes, new_priorities)
                if update_step_t % target_network_update_freq == 0:
                    # Update target network periodically.
                    update_target()
                update_step_t += 1

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
