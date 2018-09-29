from policy_gradient import PGAgent
from policy_gradient.network import Network

import numpy as np
import unittest


class TestNetwork(unittest.TestCase):

    def test_predict(self):
        net = Network()
        state = [2.5] * 172
        action = net.predict(state=state)

    def test_train(self):
        net = Network()
        states = np.array([[2.5] * 172] * 100, dtype='float32')
        action = [1] + ([0] * 24)
        actions = np.array([action] * 100, dtype='float32')
        rewards = np.zeros(shape=(100,))
        rewards[-1] = 1000
        rewards[50] = 100
        rewards = PGAgent.discount_rewards(rewards=rewards, gamma=0.7)
        net.train(states=states, actions=actions, rewards=rewards)
