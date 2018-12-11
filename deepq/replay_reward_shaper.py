# Author: Mikita Sazanovich

import os
import pickle
import numpy as np

from deepq.state_preprocessor import StatePreprocessor
from dotaenv.codes import SHAPER_STATE_PROJECT

EPS = 1e-1
K = 100


class ReplayRewardShaper:
    """
    Uses replays to parse demonstrated states and provides potentials based
    on them.
    """

    def __init__(self, replay_dir):
        self.replay_dir = replay_dir
        self.state_preprocessor = StatePreprocessor()
        self.demos = []

    def load(self):
        # Make replay selection deterministic
        demo_names = sorted(list(os.listdir(self.replay_dir)))
        for name in demo_names:
            dump_path = os.path.join(self.replay_dir, name)
            with open(dump_path, 'rb') as dump_file:
                replay = pickle.load(dump_file)
            demo = self.__process_replay(replay)
            self.demos.append(demo)
            # Only a single demo for now
            break

    def __process_replay(self, replay):
        demo = []
        for replay_step in replay:
            state, action_obs = replay_step
            state_proj = state[SHAPER_STATE_PROJECT]
            state_proc = self.state_preprocessor.process(state_proj)
            if not demo or np.linalg.norm(demo[len(demo) - 1] - state_proc) > 0:
                demo.append(state_proc)
        return demo

    def get_state_potential(self, state):
        """ Returns the state potential that is a float from [0; K).

        It represents the completion of the demo episode.
        """
        if len(state) < len(SHAPER_STATE_PROJECT):
            return 0.0
        state = state[SHAPER_STATE_PROJECT]
        for demo in self.demos:
            for i in reversed(range(len(demo))):
                diff = np.linalg.norm(demo[i] - state)
                if diff < EPS:
                    return K*(i/len(demo))
        return 0


def main():
    replay_processor = ReplayRewardShaper('../replays')
    replay_processor.load()


if __name__ == '__main__':
    main()
