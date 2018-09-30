import unittest

import numpy as np
from matplotlib import pyplot as plt

from policy_gradient.agent import PGAgent


class TestPGAgent(unittest.TestCase):

    def test_discount_rewards(self):
        data = np.array([0, 1, 1], dtype='float32')
        expected = np.array([0.75, 1.5, 1], dtype='float32')
        data = PGAgent.discount_rewards(rewards=data, gamma=0.5)
        self.assertTrue(np.allclose(data, expected))

    def test_normalize_rewards(self):
        data = np.array([0, 1, 1], dtype='float32')
        # expected = np.array([0.75, 1.5, 1], dtype='float32')
        data = PGAgent.discount_rewards(rewards=data, gamma=0.5)
        print(PGAgent.normalize_rewards(rewards=data))
        # self.assertTrue(np.allclose(data, expected))

    def test_loss(self):
        data = np.array([-117.1] * 500, dtype=float)
        disc = PGAgent.discount_rewards(rewards=data, gamma=0.99)
        norm = PGAgent.normalize_rewards(rewards=disc)
        plt.hist(norm)
        plt.show()

