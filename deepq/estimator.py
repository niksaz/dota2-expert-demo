# Author: Mikita Sazanovich

import os
import tensorflow as tf


class Estimator:
    """Q-Value estimation neural network.

    Used for both Q-Value estimation and the target network.
    """

    def __init__(
            self,
            state_space,
            action_space,
            scope="estimator",
            summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model(state_space, action_space)
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self, state_space, action_space):
        # Input
        self.X = tf.placeholder(shape=[None, state_space], dtype=tf.float32, name="X")
        # The TD value
        self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
        # Selected action index
        self.action_ind = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        layer_shape = 16

        # Network
        fc1 = tf.layers.dense(inputs=self.X, units=layer_shape, activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=layer_shape, activation=tf.nn.relu)
        fc3 = tf.layers.dense(inputs=fc2, units=action_space, activation=None)

        self.predictions = fc3

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.X)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.action_ind
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.Y, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer parameters are taken from DQN paper (V. Mnih 2015)
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=0.00025,
            momentum=0.95,
            decay=0.0,
            epsilon=0.01,)
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=tf.train.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))])

    def predict(self, sess, X):
        feed_dict = {self.X: X}
        return sess.run(self.predictions, feed_dict=feed_dict)

    def update(self, sess, X, actions, targets):
        feed_dict = {self.X: X, self.Y: targets, self.action_ind: actions}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
