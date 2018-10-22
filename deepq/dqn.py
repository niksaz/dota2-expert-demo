# Author: Mikita Sazanovich

import itertools
import os
import random
import pickle
import sys
from collections import namedtuple, deque

import numpy as np
import tensorflow as tf

sys.path.append('../')

from deepq import ReplayRewardShaper, Estimator, StatePreprocessor
from dotaenv import DotaEnvironment

STATE_SPACE = 3
ACTION_SPACE = 16


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:

    def __init__(self, replay_memory_size, save_dir):
        # The replay memory
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.dump_path = os.path.join(save_dir, 'replay_buffer.pickle')

    def push(self, transition):
        if transition is None:
            return
        self.replay_memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.replay_memory, batch_size)

    def save_buffer(self):
        print('saving to', self.dump_path)
        with open(self.dump_path, 'wb') as dump_file:
            pickle.dump(self.replay_memory, dump_file)

    def load_buffer(self):
        if os.path.exists(self.dump_path):
            print('loading from', self.dump_path)
            with open(self.dump_path, 'rb') as dump_file:
                self.replay_memory = pickle.load(dump_file)


class TransitionBuilder:

    def __init__(self, discount_factor, replay_processor, action_oracle):
        self.discount_factor = discount_factor
        self.replay_processor = replay_processor
        self.action_oracle = action_oracle

    def build(self, state, action, reward, next_state, done):
        # Discard invalid transitions
        if len(state) != STATE_SPACE or len(next_state) != STATE_SPACE:
            return None
        potential = self.replay_processor.get_potential(state, action)
        next_action = self.action_oracle(next_state)
        next_potential = self.replay_processor.get_potential(next_state, next_action)
        reward += self.discount_factor * next_potential - potential
        return Transition(state, action, reward, next_state, done)


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


def make_epsilon_greedy_policy(estimator, replay_processor, nA=ACTION_SPACE):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        for action in range(nA):
            q_values[action] += replay_processor.get_potential(observation, action)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def populate_replay_buffer(replay_buffer, transition_builder, action_sampler, env):
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
        replay_buffer.push(transition_builder.build(state, action, reward, next_state, done))
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

    replay_buffer = ReplayBuffer(replay_memory_size, save_dir=experiment_dir)

    reward_shaper = ReplayRewardShaper('../replays/')
    reward_shaper.load()

    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator, reward_shaper, ACTION_SPACE)

    transition_builder = TransitionBuilder(
        discount_factor=discount_factor,
        replay_processor=reward_shaper,
        action_oracle=lambda state: np.argmax(policy(sess, state, 0.0)))

    # Populate the replay memory with initial experience
    action_sampler = lambda state: policy(
        sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
    populate_replay_buffer(replay_buffer, transition_builder, action_sampler, env)

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
            replay_buffer.push(
                transition_builder.build(state, action, reward, next_state, done))

            # Sample a minibatch from the replay memory
            samples = replay_buffer.sample(batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets (Double DQN)
            q_values_next = q_estimator.predict(sess, next_states_batch)

            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            print("\rStep {} ({}/{}) @ Episode {}, loss: {}".format(
                    t, total_t, num_steps, i_episode, loss), end="")
            sys.stdout.flush()

            state = next_state
            total_t += 1


def main():
    env = DotaEnvironment()

    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/biased-greedy-rms-prop")

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
            num_steps=15000,
            replay_memory_size=5000,
            update_target_estimator_every=1000,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=10000,
            restore=False)


if __name__ == "__main__":
    main()
