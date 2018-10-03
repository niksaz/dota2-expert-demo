# Author: Mikita Sazanovich

import numpy as np
import tensorflow as tf

from policy_gradient.network import Network
from policy_gradient.replay_buffer import ReplayBuffer


def main():
    input_shape = 2
    output_shape = 16
    replay_buffer = ReplayBuffer()
    network = Network(input_shape=input_shape,
                      output_shape=output_shape,
                      restore=True)

    tvars = tf.trainable_variables()
    tvars_vals = network.session.run(tvars)

    for var, val in zip(tvars, tvars_vals):
        print(var.name,
              val)  # Prints the name of the variable alongside its value.

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
