import logging

import tensorflow as tf
import numpy as np

logger = logging.getLogger('DotaRL.Network')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Network:
    """
    MLP classifier for predicting actions by given states.
    """
    __slots__ = ('predict_op',
                 'train_op',
                 'states',
                 'actions',
                 'rewards',
                 'loss',
                 'session',
                 'saver')

    def __init__(self, input_shape=172, output_shape=25, learning_rate=0.01,
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

    def build(self, input_shape=172, output_shape=25, learning_rate=0.01, layer_shape=80):
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

        # network
        layer1 = tf.layers.dense(inputs=input_layer, units=layer_shape, activation=tf.nn.relu)
        layer2 = tf.layers.dense(inputs=layer1, units=layer_shape, activation=tf.nn.relu)
        # layer3 = tf.layers.dense(inputs=layer2, units=layer_shape, activation=tf.nn.selu)
        # layer4 = tf.layers.dense(inputs=layer3, units=layer_shape, activation=tf.nn.selu)

        # output TODO activation
        logits = tf.layers.dense(inputs=layer2, units=output_shape, activation=tf.nn.relu)

        # loss
        ce = tf.losses.softmax_cross_entropy(onehot_labels=onehot_actions, logits=logits)
        loss = tf.multiply(ce, normalized_rewards)
        self.loss = loss

        # predict operation
        self.predict_op = tf.nn.softmax(logits=logits)

        # train operation
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(loss=loss)

    def train(self, states, actions, rewards):
        """
        Train network with given batch of rollout data.
        Batch size is considered as states.shape[0]

        :param states: np array of shape (batch_size, input_shape)
        :param actions:  np array of shape (batch_size, output_shape)
        :param rewards: normalized discounted rewards np.array of shape (batch_size, )
        """
        # TODO TEMP SLICE
        self.session.run(self.train_op, feed_dict={self.states: states[:, :2], self.actions: actions,
                                                   self.rewards: rewards})
        loss = self.session.run(self.loss, feed_dict={self.states: states[:, :2], self.actions: actions,
                                               self.rewards: rewards})
        #logger.debug(loss)
        self.saver.save(self.session, 'saved_model/model.ckpt')

    def predict(self, state):
        """
        Predict action for single given state.

        :param state: given state
        :return: predicted action (number)
        """
        # TODO TEMP SLICE
        return np.argmax(self.session.run(self.predict_op, feed_dict={self.states: np.array([state[:2]])}))
