# Author: Mikita Sazanovich

import os
import pickle
from collections import deque

import numpy as np
from namedlist import namedlist

from dotaenv.codes import STATE_DIM

MAX_PRIORITY = 1
EPS_PRIORITY = 1e-9

Transition = namedlist(
    'Transition',
    ['state', 'action', 'next_state', 'done', 'reward', 'priority'])


class PrioritizedReplayBuffer:
    """Reference paper: https://arxiv.org/pdf/1511.05952.pdf.
    """

    def __init__(self,
                 replay_memory_size,
                 total_steps,
                 reward_shaper,
                 discount_factor,
                 save_dir,
                 alpha=0.6,
                 beta0=0.4):
        """Initializes the replay buffer and caps the memory size to replay_memory_size.
        """
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.total_steps = total_steps
        self.reward_shaper = reward_shaper
        self.discount_factor = discount_factor
        self.dump_path = os.path.join(save_dir, 'replay_buffer.pickle')
        self.alpha = alpha
        self.beta0 = beta0

    def push(self, state, action, next_state, done, reward):
        """ Pushes the transition into memory with MAX_PRIORITY.

        If the starting or resulting states are incorrect the transition is
        omitted.
        """
        if len(state) != STATE_DIM or len(next_state) != STATE_DIM:
            return None
        # Potential based-reward shaping
        reward += (self.discount_factor*self.reward_shaper.get_state_potential(next_state) -
                   self.reward_shaper.get_state_potential(state))
        transition = Transition(state, action, next_state, done, reward, MAX_PRIORITY)
        self.replay_memory.append(transition)

    def sample(self, batch_size, step):
        """Samples the batch according to priorities.

        Returns a tuple of (samples, weights, idx).
        """
        N = len(self.replay_memory)
        # Transition sampling probabilities.
        p = np.zeros(N)
        for i in range(N):
            p[i] = self.replay_memory[i].priority ** self.alpha
        p /= p.sum()
        # Indices of samples.
        idx = np.random.choice(N, batch_size, replace=False, p=p).tolist()
        samples = [self.replay_memory[id] for id in idx]
        # Linearly annealing importance-sampling exponent.
        beta = self.beta0 + (1 - self.beta0) * (step / self.total_steps)
        # Importance-sampling weights.
        weights = (N * p[idx]) ** (-beta)
        # Normalize weights.
        weights = weights / np.max(weights)
        return samples, weights, idx

    def update_priorities(self, idx, deltas):
        for index, delta in zip(idx, deltas):
            priority = min(MAX_PRIORITY, delta + EPS_PRIORITY)
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
