# Author: Mikita Sazanovich

import os
import pickle
import numpy as np
import math

from deepq import StatePreprocessor

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
        # Remove consecutive duplicates
        states = []
        for entry in replay:
            # x,y,facing,enemy_x,enemy_y,creep_x,creep_y
            entry = entry[[0,1,2,11,12,19,20, 83,84,85]]
            if not states or np.any(states[len(states) - 1] != entry):
                states.append(entry)
        demo = []
        for i in range(1, len(states)):
            prev_state = states[i - 1][:-3]
            next_state = states[i][:-3]

            if states[i - 1][-3] == 1:
                # attack the nearest creep
                action = 16
            elif states[i - 1][-2] == 1:
                # attack the enemy hero
                action = 17
            elif states[i - 1][-1] == 1:
                # attack the enemy tower
                action = 18
            else:
                diff = next_state - prev_state
                if np.all(diff == 0):
                    continue
                angle_pi = math.atan2(diff[1], diff[0])
                if angle_pi < 0:
                    angle_pi += 2 * math.pi
                degrees = angle_pi / math.pi * 180
                action = round(degrees / 22.5) % 16

            demo.append((
                self.state_preprocessor.process(prev_state),
                action,
                self.state_preprocessor.process(next_state)))
        return demo

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


def main():
    replay_processor = ReplayRewardShaper('../replays')
    replay_processor.load()


if __name__ == '__main__':
    main()
