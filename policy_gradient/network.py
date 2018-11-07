import logging

import tensorflow as tf
import numpy as np

logger = logging.getLogger('DotaRL.Network')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Network:
    """
    Policy gradient network for predicting actions by a given state.
    """
    __slots__ = ('predict_op',
                 'train_op',
                 'states',
                 'actions',
                 'rewards',
                 'loss',
                 'session',
                 'saver')

    def __init__(self, input_shape, output_shape, learning_rate=0.01,
                 restore=False):
        self.predict_op = None
        self.train_op = None
        self.states = None
        self.actions = None
        self.rewards = None
        self.loss = None
        self.build(input_shape=input_shape, output_shape=output_shape, learning_rate=learning_rate)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # initialize saver and restore if needed
        self.saver = tf.train.Saver()
        if restore:
            self.saver.restore(self.session, 'saved_model/model.ckpt')

    def build(self, input_shape, output_shape, learning_rate=0.01,
              layer_shape=20):
        """
        Build action classifier network for policy gradient algorithm.

        :param input_shape: state shape
        :param output_shape: number of actions
        :param learning_rate: learning rate
        :param layer_shape: inner layer shape
        """
        onehot_actions = tf.placeholder(dtype='float32', shape=(None, output_shape), name='y')
        input_layer = tf.placeholder(dtype='float32', shape=(None, input_shape), name='x')
        normalized_rewards = tf.placeholder(dtype='float32', shape=(None,), name='rewards')

        self.states = input_layer
        self.actions = onehot_actions
        self.rewards = normalized_rewards

        in_layer = tf.Print(input_layer, [input_layer], message='input_layer', summarize=input_shape)

        # network
        fc1 = tf.layers.dense(inputs=in_layer, units=layer_shape, activation=tf.nn.relu)
        fc1_print = tf.Print(fc1, [fc1], message='fc1', summarize=layer_shape)

        fc2 = tf.layers.dense(inputs=fc1_print, units=layer_shape, activation=tf.nn.relu)
        fc2_print = tf.Print(fc2, [fc2], message='fc2', summarize=layer_shape)

        fc3 = tf.layers.dense(inputs=fc2_print, units=output_shape, activation=None)
        fc3_print = tf.Print(fc3, [fc3], message='fc3', summarize=output_shape)

        # predict operation
        self.predict_op = tf.nn.softmax(logits=fc3_print)

        # loss function
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3_print,
                                                                  labels=self.actions)
        self.loss = tf.reduce_mean(neg_log_prob * self.rewards)

        # train operation
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss=self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def train(self, states, actions, rewards):
        """
        Train network with given batch of rollout data.
        Batch size is considered as states.shape[0]

        :param states: np array of shape (batch_size, input_shape)
        :param actions:  np array of shape (batch_size, output_shape)
        :param rewards: normalized discounted rewards np.array of shape (batch_size, )
        """
        var_dict = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards
        }
        self.session.run(self.train_op, feed_dict=var_dict)
        loss = self.session.run(self.loss, feed_dict=var_dict)
        logger.debug('Loss:')
        logger.debug(loss)
        self.saver.save(self.session, 'saved_model/model.ckpt')

    def predict(self, state):
        """
        Predict an action for a given state.

        :param state: a given state
        :return: the predicted action to take
        """
        var_dict = {self.states: np.array([state])}
        return np.argmax(self.session.run(self.predict_op, feed_dict=var_dict))
