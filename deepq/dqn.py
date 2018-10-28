# Author: Mikita Sazanovich

import itertools
import os
import pickle
import sys
import argparse
from collections import deque
from namedlist import namedlist

import numpy as np
import tensorflow as tf

sys.path.append('../')

from deepq import ReplayRewardShaper, Estimator, StatePreprocessor
from dotaenv import DotaEnvironment
from dotaenv.codes import ATTACK_TOWER, STATE_PROJECT

STATE_SPACE = len(STATE_PROJECT)
ACTION_SPACE = ATTACK_TOWER + 1
MAX_PRIORITY = 10
EPS_PRIORITY = 1e-9

Transition = namedlist(
    'Transition',
    ['state', 'action', 'next_state', 'done', 'reward', 'priority'])


class PrioritizedReplayBuffer:
    """Reference paper: https://arxiv.org/pdf/1511.05952.pdf.
    """

    def __init__(self, replay_memory_size, alpha, beta0, save_dir):
        """Initializes the replay buffer and caps the memory size to replay_memory_size.
        """
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.alpha = alpha
        self.beta0 = beta0
        self.dump_path = os.path.join(save_dir, 'replay_buffer.pickle')

    def push(self, state, action, next_state, done, reward):
        """ Pushes the transition into memory with MAX_PRIORITY.

        If the starting or resulting states are incorrect the transition is
        omitted.
        """
        if len(state) != STATE_SPACE or len(next_state) != STATE_SPACE:
            return None
        transition = Transition(state, action, next_state, done, reward, MAX_PRIORITY)
        self.replay_memory.append(transition)

    def sample(self, batch_size):
        """Samples the batch according to priorities.

        Returns a tuple of (transitions, indices).
        """
        buffer_size = len(self.replay_memory)
        p = np.zeros(buffer_size)
        for i in range(buffer_size):
            p[i] = self.replay_memory[i].priority ** self.alpha
        p /= p.sum()
        idx = np.random.choice(buffer_size, batch_size, replace=False, p=p).tolist()
        samples = [self.replay_memory[id] for id in idx]
        return samples, idx

    def update_priorities(self, idx, priorities):
        for index, priority in zip(idx, priorities):
            self.replay_memory[index].priority = min(MAX_PRIORITY, priority)

    def save_buffer(self):
        print('saving to', self.dump_path)
        with open(self.dump_path, 'wb') as dump_file:
            pickle.dump(self.replay_memory, dump_file)

    def load_buffer(self):
        if os.path.exists(self.dump_path):
            print('loading from', self.dump_path)
            with open(self.dump_path, 'rb') as dump_file:
                self.replay_memory = pickle.load(dump_file)


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the parameters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, reward_shaper, acts):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        reward_shaper: Reward shaper for a (state, action) pair.
        acts: Number of actions in the environment.
    Returns:
        A function that takes the (sess, state, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(sess, state, epsilon):
        A = np.ones(acts, dtype=float) * epsilon / acts
        q_values = estimator.predict(sess, np.expand_dims(state, 0))[0]
        for action in range(acts):
            q_values[action] += reward_shaper.get_potential(state, action)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def populate_replay_buffer(replay_buffer, action_sampler, env):
    print("Populating replay memory...")
    state = env.reset()
    state = StatePreprocessor.process(state)
    done = False
    for t in itertools.count():
        if done:
            break
        action_probs = action_sampler(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        print("Step {step} state: {state}, action: {action}.".format(step=t, state=state, action=action))
        next_state, reward, done = env.execute(action=action)
        next_state = StatePreprocessor.process(next_state)
        replay_buffer.push(state, action, next_state, done, reward)
        state = next_state


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_steps,
                    experiment_dir,
                    replay_memory_size=5000,
                    update_target_estimator_every=500,
                    discount_factor=0.999,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=10000,
                    update_q_values_every=4,
                    batch_size=32,
                    restore=True):

    reward_dir = os.path.join(experiment_dir, "rewards")
    if not os.path.exists(reward_dir):
        os.makedirs(reward_dir)
    reward_writer = tf.summary.FileWriter(reward_dir)

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    saver = tf.train.Saver()
    if restore:
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.train.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    replay_buffer = PrioritizedReplayBuffer(
        replay_memory_size,
        alpha=0.6,
        beta0=0.4,
        save_dir=experiment_dir)

    reward_shaper = ReplayRewardShaper('../replays/')
    reward_shaper.load()

    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator, reward_shaper, ACTION_SPACE)

    # Populate the replay memory with initial experience
    action_sampler = lambda state: policy(
        sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
    populate_replay_buffer(replay_buffer, action_sampler, env)

    print('Training is starting...')
    # Training the agent
    for i_episode in itertools.count():
        episode_reward = 0
        multiplier = 1

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset()
        state = StatePreprocessor.process(state)
        done = False

        # One step in the environment
        for t in itertools.count():
            if total_t >= num_steps:
                return

            eps = epsilons[min(total_t, epsilon_decay_steps-1)]

            if done or len(state) != STATE_SPACE:
                print("Finished episode with reward", episode_reward)
                summary = tf.Summary(value=[tf.Summary.Value(tag="rewards", simple_value=episode_reward)])
                reward_writer.add_summary(summary, i_episode)
                summary = tf.Summary(value=[tf.Summary.Value(tag="eps", simple_value=eps)])
                reward_writer.add_summary(summary, i_episode)
                break

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Take a step
            action_probs = policy(sess, state, eps)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.execute(action=action)
            next_state = StatePreprocessor.process(next_state)

            episode_reward += reward * multiplier
            multiplier *= discount_factor

            # Save transition to replay memory
            replay_buffer.push(state, action, next_state, done, reward)

            if total_t % update_q_values_every == 0:
                # Sample a minibatch from the replay memory
                samples, idx = replay_buffer.sample(batch_size)
                states, actions, next_states, dones, rewards, _ = map(np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                next_q_values = q_estimator.predict(sess, next_states)
                for i in range(batch_size):
                    for action in range(ACTION_SPACE):
                        next_q_values[i][action] += reward_shaper.get_potential(next_states[i], action)
                next_actions = np.argmax(next_q_values, axis=1)

                next_q_values_target = target_estimator.predict(sess, next_states)
                not_dones = np.invert(dones).astype(np.float32)

                targets = (
                    rewards
                    + discount_factor * reward_shaper.get_potentials(next_states, next_actions)
                    - reward_shaper.get_potentials(states, actions)
                    + discount_factor * not_dones * next_q_values_target[np.arange(batch_size), next_actions])

                # Perform gradient descent update
                predictions = q_estimator.update(sess, states, actions, targets)

                # Update transition priorities
                priors = np.abs(predictions - targets) + EPS_PRIORITY
                replay_buffer.update_priorities(idx, priors)

            print("\rStep {}, episode {} ({}/{})".format(t, i_episode, total_t, num_steps), end="\t")
            sys.stdout.flush()

            state = next_state
            total_t += 1


def main():
    parser = argparse.ArgumentParser(description='Trains the agent by DQN')
    parser.add_argument('experiment', help='specifies the experiment name')
    args = parser.parse_args()

    env = DotaEnvironment()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.join(os.path.abspath("./experiments/"), args.experiment)

    tf.reset_default_graph()
    # Create a global step variable
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Create estimators
    q_estimator = Estimator(
        STATE_SPACE,
        ACTION_SPACE,
        scope="q",
        summaries_dir=experiment_dir)
    target_estimator = Estimator(
        STATE_SPACE,
        ACTION_SPACE,
        scope="target_q")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        deep_q_learning(
            sess=sess,
            env=env,
            q_estimator=q_estimator,
            target_estimator=target_estimator,
            experiment_dir=experiment_dir,
            num_steps=150000,
            replay_memory_size=10000,
            epsilon_decay_steps=1,
            epsilon_start=0.1,
            epsilon_end=0.1,
            update_target_estimator_every=1000,
            update_q_values_every=4,
            batch_size=32,
            restore=False)

    env.close()


if __name__ == "__main__":
    main()
