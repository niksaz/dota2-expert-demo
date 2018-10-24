# Author: Mikita Sazanovich

import numpy as np


class StatePreprocessor:

    @staticmethod
    def process(state):
        state = np.copy(state)
        if len(state) > 0:
            state[0] = StatePreprocessor._process_coordinate(state[0])
        if len(state) > 1:
            state[1] = StatePreprocessor._process_coordinate(state[1])
        if len(state) > 2:
            state[2] = StatePreprocessor._process_facing(state[2])
        if len(state) > 3:
            state[3] = StatePreprocessor._process_coordinate(state[3])
        if len(state) > 4:
            state[4] = StatePreprocessor._process_coordinate(state[4])
        if len(state) > 5:
            state[5] = StatePreprocessor._process_coordinate(state[5])
        if len(state) > 6:
            state[6] = StatePreprocessor._process_coordinate(state[6])
        return state

    @staticmethod
    def _process_coordinate(coord):
        # Normalize a coordinate from [-8288; 8288] to [-1, 1]
        return coord / 8288

    @staticmethod
    def _process_facing(angle):
        # Normalize facing from [0; 360] to [-1, 1]
        return (angle - 180) / 180
