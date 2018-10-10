# Author: Mikita Sazanovich

import tensorflow as tf
import numpy as np
import os
import sys
import itertools
import random
from collections import namedtuple

sys.path.append('../')

from dotaenv import DotaEnvironment

state_space = 3
action_space = 16


class Estimator:
    """Q-Value estimation neural network.

    Used for both Q-Value estimation and the target network.
    """

    def __init__(self, scope="estimator"):
        self.scope = scope
        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        # Input
        self.X = tf.placeholder(shape=[None, state_space], dtype=tf.float32, name="X")
        # The TD value
        self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
        # Selected action index
        self.action_ind = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        layer_shape = 16
        batch_size = tf.shape(self.X)[0]

        # in_layer = tf.Print(self.X, [self.X], message="input_layer", summarize=state_space)

        # Network
        fc1 = tf.layers.dense(inputs=self.X, units=layer_shape,
                              activation=tf.nn.relu)
        # fc1_print = tf.Print(fc1, [fc1], message="fc1", summarize=layer_shape)

        fc2 = tf.layers.dense(inputs=fc1, units=action_space, activation=None)
        # fc2_print = tf.Print(fc2, [fc2], message="fc2", summarize=action_space)

        self.predictions = fc2

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.action_ind
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.Y, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, X):
        feed_dict = {self.X: X}
        return sess.run(self.predictions, feed_dict=feed_dict)

    def update(self, sess, X, actions, targets):
        feed_dict = {self.X: X, self.Y: targets, self.action_ind: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        print("Loss {}".format(loss))
        return loss


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
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


def make_epsilon_greedy_policy(estimator, nA=action_space):
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
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=500,
                    update_target_estimator_every=1000,
                    discount_factor=0.999,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=10000,
                    batch_size=32):

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator, action_space)

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done = env.execute(action=action)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
        else:
            state = next_state
        print("Step {step} state: {state}, action: {action}.".format(step=i, rew=reward, action=action, state=state))

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset()
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.execute(action=action)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets (Double DQN)
            try:
                q_values_next = q_estimator.predict(sess, next_states_batch)
            except:
                print("CAUTHG!!!!")
                print(samples)
                print(states_batch)
                print(action_batch)
                print(reward_batch)
                print(next_states_batch)
                print(done_batch)
                exit(1)

            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_t += 1


def main():
    env = DotaEnvironment()

    tf.reset_default_graph()
    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/")

    # Create a global step variable
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Create estimators
    q_estimator = Estimator(scope="q")
    target_estimator = Estimator(scope="target_q")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        deep_q_learning(
            sess=sess,
            env=env,
            q_estimator=q_estimator,
            target_estimator=target_estimator,
            experiment_dir=experiment_dir,
            num_episodes=200)


if __name__ == "__main__":
    main()
