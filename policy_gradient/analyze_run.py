# Author: Mikita Sazanovich

import numpy as np
import tensorflow as tf
import logging

from policy_gradient.network import Network
from policy_gradient.replay_buffer import ReplayBuffer

logger = logging.getLogger('DotaRL.PGAgent')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def print_network_weights(network):
    tvars = tf.trainable_variables()
    tvars_vals = network.session.run(tvars)

    for var, val in zip(tvars, tvars_vals):
        logger.debug(var.name)
        logger.debug(val)


def main():
    input_shape = 3
    output_shape = 16
    replay_buffer = ReplayBuffer()
    network = Network(input_shape=input_shape,
                      output_shape=output_shape,
                      restore=True)
    print_network_weights(network)

    print("Replays:")
    replay_buffer.load_data()
    N = len(replay_buffer.data)
    print("N: ", N)
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        state, action, reward = replay_buffer.data[i]
        x[i] = state[0]
        y[i] = state[1]

    print("x.mean x.std", x.mean(), x.std())
    print("y.mean y.std", y.mean(), y.std())

    exit(1)


if __name__ == '__main__':
    main()
