# Author: Mikita Sazanovich

import json
import math
import os
import pickle
import matplotlib.pyplot as plt
from collections import namedtuple
from abc import ABC, abstractmethod

import numpy as np

from deepq.state_preprocessor import StatePreprocessor
from dotaenv.codes import SHAPER_STATE_PROJECT, SHAPER_STATE_DIM, STATE_DIM, \
    ACTIONS_TOTAL

DemoStateActionPair = namedtuple('DemoStateActionPair', ['id', 'state', 'action'])


class AbstractRewardShaper(ABC):

    @abstractmethod
    def __init__(self, replay_dir):
        self.replay_dir = replay_dir
        self.state_preprocessor = StatePreprocessor
        self.demos = []

    @abstractmethod
    def load(self):
        # Making the limiting of the number of replays deterministic
        demo_names = sorted(list(os.listdir(self.replay_dir)))
        for name in demo_names:
            dump_path = os.path.join(self.replay_dir, name)
            with open(dump_path, 'rb') as dump_file:
                dumped_replay = pickle.load(dump_file)
            demo = self.process_replay(dumped_replay)
            self.demos.append(demo)

    @abstractmethod
    def process_replay(self, dumped_replay):
        pass


class StatePotentialRewardShaper(AbstractRewardShaper):
    """
    Uses replays to parse demonstrated states and provides potentials based
    on them.
    """
    CLOSE_TO_STATE_EPS = 1e-1
    K = 100

    def __init__(self, replay_dir):
        super(StatePotentialRewardShaper, self).__init__(replay_dir)

    def load(self):
        super(StatePotentialRewardShaper, self).load()
        # Experimenting with the different number of replays
        replays_to_leave = 3
        self.demos = self.demos[:replays_to_leave]
        assert len(self.demos) == replays_to_leave

    def process_replay(self, replay):
        demo = []
        for replay_step in replay:
            if len(replay_step) == 0:
                continue
            state = replay_step
            state_proj = state[SHAPER_STATE_PROJECT]
            state_proc = self.state_preprocessor.process(state_proj)
            if not demo or np.linalg.norm(demo[len(demo) - 1] - state_proc) > 0:
                demo.append(state_proc)
        return demo

    def get_state_potential(self, state):
        """ Returns the state potential that is a float from [0; K).

        It represents the maximum completion of the episode across replays.
        """
        if len(state) < SHAPER_STATE_DIM:
            return 0.0
        max_potent = 0.0
        state = state[SHAPER_STATE_PROJECT]
        for demo in self.demos:
            for i in range(len(demo)):
                diff = np.linalg.norm(demo[i] - state)
                if diff < StatePotentialRewardShaper.CLOSE_TO_STATE_EPS:
                    max_potent = max(max_potent, StatePotentialRewardShaper.K * ((i + 1) / len(demo)))
        return max_potent


class ActionAdviceRewardShaper(AbstractRewardShaper):
    SIGMA = 0.2 * np.identity(STATE_DIM)
    SIGMA[0][0] = 1.0
    SIGMA[1][1] = 1.0
    SIGMA[2][2] = 1.0
    K = 10

    FILTER_THRESHOLD = 0.95

    @staticmethod
    def get_states_similarity(state1, state2):
        diff = state1 - state2
        value = math.e ** (-1 / 2 * diff.dot(ActionAdviceRewardShaper.SIGMA).dot(diff))
        return value

    def __init__(self, replay_dir):
        super(ActionAdviceRewardShaper, self).__init__(replay_dir)
        self.state_actions = []

    def load(self):
        self.demos.append([])  # Imaginary demo to count non-demoed actions
        filenames = os.listdir(self.replay_dir)
        filenames = sorted(filenames)
        for filename in filenames:
            filepath = os.path.join(self.replay_dir, filename)
            file = open(filepath, 'r')
            lines = file.readlines()
            file.close()
            demo = self.process_replay(lines)
            self.demos.append(demo)
            print('Loaded demo from {}. Its length is {}'.format(filepath, len(demo)))
        print('Total number of state-action pairs:', sum(map(len, self.demos)))

    def process_replay(self, replay_lines):
        last_action = 0
        demo = []
        for line in replay_lines:
            state_action_pair = json.loads(line)
            state = state_action_pair['state']
            action = state_action_pair['action']
            if action >= ACTIONS_TOTAL:
                continue
            vector_state = np.zeros(18, dtype=np.float32)
            vector_state[0] = last_action / (ACTIONS_TOTAL - 1.0)
            vector_state[1:12] = state['hero_info']
            vector_state[12:] = state['enemy_info']
            demo.append((vector_state, action))
            last_action = action
        return demo

    def filter(self):
        self.state_actions = []
        for ind, demo in enumerate(self.demos):
            for demo_state, demo_action in demo:
                similar = False
                for (_, state, action) in self.state_actions:
                    sim = ActionAdviceRewardShaper.get_states_similarity(
                        demo_state, state)
                    if sim > ActionAdviceRewardShaper.FILTER_THRESHOLD:
                        similar = True
                        break
                if not similar:
                    self.state_actions.append(DemoStateActionPair(ind, demo_state, demo_action))
        print('State-action pairs after filtering:', len(self.state_actions))

    def get_action_potentials(self, state, return_demo_indexes=False):
        potentials = np.zeros(ACTIONS_TOTAL, dtype=np.float32)
        if return_demo_indexes:
            demo_indexes = np.zeros(ACTIONS_TOTAL, dtype=int)
        for demo_ind, demo_state, demo_action in self.state_actions:
            potential = ActionAdviceRewardShaper.get_states_similarity(state, demo_state)
            potential *= ActionAdviceRewardShaper.K
            if return_demo_indexes:
                if potential > potentials[demo_action]:
                    potentials[demo_action] = potential
                    demo_indexes[demo_action] = demo_ind
            else:
                potentials[demo_action] = max(potentials[demo_action], potential)
        if return_demo_indexes:
            return potentials, demo_indexes
        else:
            return potentials


def plot_distance_distrib(demo):
    dsts = []
    last_state = None
    for state, _ in demo:
        if last_state is not None:
            dsts.append(ActionAdviceRewardShaper.get_states_similarity(last_state, state))
        last_state = state

    hist, bins = np.histogram(dsts, range=[0, 1], bins=20)
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=bins[1]-bins[0])
    plt.show()


def main():
    reward_shaper = ActionAdviceRewardShaper('../completed-observations')
    reward_shaper.load()
    reward_shaper.filter()
    for ind, state, action in reward_shaper.state_actions:
        print(ind, state, action)
        action_potentials = reward_shaper.get_action_potentials(state)
        print('action potentials are:', action_potentials)
        action_potentials, action_demos = reward_shaper.get_action_potentials(state, return_demo_indexes=True)
        print('action potentials are:', action_potentials)
        print('action demos are:', action_demos)


if __name__ == '__main__':
    main()
