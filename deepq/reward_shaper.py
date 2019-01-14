# Author: Mikita Sazanovich

import os
import pickle
import numpy as np
import math
from abc import ABC, abstractmethod

from deepq.state_preprocessor import StatePreprocessor
from dotaenv.codes import ATTACK_CREEP, ATTACK_HERO, ATTACK_TOWER, \
    SHAPER_STATE_PROJECT, SHAPER_STATE_DIM, STATE_PROJECT, STATE_DIM, \
    MOVE_ACTIONS_TOTAL, ACTIONS_TOTAL


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
    K = 10

    def __init__(self, replay_dir):
        super(ActionAdviceRewardShaper, self).__init__(replay_dir)

    def load(self):
        super(ActionAdviceRewardShaper, self).load()
        print('Loaded %d action replays'.format(len(self.demos)))

    def process_replay(self, dumped_replay):
        demo = []
        for (state, info) in dumped_replay:
            if len(state) == 0:
                continue
            state = state[STATE_PROJECT]
            state = self.state_preprocessor.process(state)

            if info[2] != -1:  # The target is an enemy creep
                action = ATTACK_CREEP
            elif info[3] != -1:  # The target is an enemy hero
                action = ATTACK_HERO
            elif info[4] != -1:  # The target is an enemy tower
                action = ATTACK_TOWER
            else:
                # Maybe the agent is moving?
                diff = np.array(info[:2], dtype=np.float32) - state[:2]
                if np.linalg.norm(diff) == 0:
                    # position did not change; skip transition
                    continue
                angle_pi = math.atan2(diff[1], diff[0])
                if angle_pi < 0:
                    angle_pi += 2 * math.pi
                degrees = angle_pi / math.pi * 180
                action = round(degrees / (360 / MOVE_ACTIONS_TOTAL)) % MOVE_ACTIONS_TOTAL
            demo.append((state, action))
        filtered = []
        last_act = None
        cnt = 0
        for state, action in demo:
            if last_act is None or action != last_act or cnt >= 30:
                filtered.append((state, action))
                cnt = 0
            else:
                cnt += 1
            last_act = action
        return filtered

    def get_action_potentials(self, states):
        potentials = np.zeros((len(states), ACTIONS_TOTAL), dtype=np.float32)
        for idx, state in enumerate(states):
            for demo in self.demos:
                for demo_state, demo_action in demo:
                    diff = state - demo_state
                    value = ActionAdviceRewardShaper.K * \
                            math.e ** (-1 / 2 * diff.dot(ActionAdviceRewardShaper.SIGMA).dot(diff))
                    potentials[idx][demo_action] = max(potentials[idx][demo_action], value)
        return potentials


def main():
    reward_shaper = ActionAdviceRewardShaper('replays-action/')
    reward_shaper.load()
    for demo in reward_shaper.demos:
        for (state, action) in demo:
            print(state, action)
            print('action potentials are:', reward_shaper.get_action_potentials([state]))
        print('Total number is', len(demo))


if __name__ == '__main__':
    main()
