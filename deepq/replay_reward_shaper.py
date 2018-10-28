# Author: Mikita Sazanovich

import os
import pickle
import numpy as np
import math

from deepq import StatePreprocessor
from dotaenv.codes import MOVES_TOTAL, ATTACK_CREEP, ATTACK_HERO, ATTACK_TOWER, STATE_PROJECT

SIGMA = 0.2 * np.identity(7)
K = 10


class ReplayRewardShaper:
    """ Provides potential-based reward shaping based on expert demonstrations.

    Uses replays to parse demonstrated state-action pairs and provides rewards
    based on them.

    Reference paper: https://www.ijcai.org/Proceedings/15/Papers/472.pdf.
    """

    def __init__(self, replay_dir):
        self.replay_dir = replay_dir
        self.state_preprocessor = StatePreprocessor()
        self.demos = []

    def load(self):
        for name in os.listdir(self.replay_dir):
            dump_path = os.path.join(self.replay_dir, name)
            with open(dump_path, 'rb') as dump_file:
                replay = pickle.load(dump_file)
            demo = self.__process_replay(replay)
            self.demos.append(demo)

    def __process_replay(self, replay):
        demo = []
        for i in range(len(replay) - 1):
            state0, action_state0 = replay[i]
            state1, _ = replay[i + 1]

            state0 = state0[STATE_PROJECT]
            state1 = state1[STATE_PROJECT]

            if action_state0[0] == 1:
                # attack the nearest creep
                action = ATTACK_CREEP
            elif action_state0[1] == 1:
                # attack the enemy hero
                action = ATTACK_HERO
            elif action_state0[2] == 1:
                # attack the enemy tower
                action = ATTACK_TOWER
            else:
                # try to move
                diff = state1[:2] - state0[:2]
                if np.linalg.norm(diff) == 0:
                    # position did not change; skip transition
                    continue
                angle_pi = math.atan2(diff[1], diff[0])
                if angle_pi < 0:
                    angle_pi += 2 * math.pi
                degrees = angle_pi / math.pi * 180
                action = round(degrees / (360 / MOVES_TOTAL)) % MOVES_TOTAL

            demo.append((
                self.state_preprocessor.process(state0),
                action,
                self.state_preprocessor.process(state1)))
        return demo[:250]

    def get_potential(self, state, action):
        best_value = 0
        for demo in self.demos:
            for demo_state, demo_action, _ in demo:
                if demo_action != action:
                    continue
                diff = state - demo_state
                value = K * math.e ** (-1/2*diff.dot(SIGMA).dot(diff))
                if value > best_value:
                    best_value = value
        return best_value

    def get_potentials(self, states, actions):
        N = states.shape[0]
        potentials = np.zeros(N)
        for i in range(N):
            potentials[i] = self.get_potential(states[i], actions[i])
        return potentials

    def get_nearest_demo(self, state):
        best_norm = None
        action = None
        for demo in self.demos:
            for demo_state, demo_action, _ in demo:
                diff = state - demo_state
                norm = np.linalg.norm(diff)
                if action is None or norm < best_norm:
                    best_norm = norm
                    action = demo_action
        return action


def main():
    replay_processor = ReplayRewardShaper('../replays')
    replay_processor.load()


if __name__ == '__main__':
    main()
