# Author: Mikita Sazanovich


class StatePreprocessor:

    @staticmethod
    def process(state):
        if len(state) > 0:
            # Normalize x position from [-8288; 8288] to [-1, 1]
            state[0] = state[0] / 8288
        if len(state) > 1:
            # Normalize y position from [-8288; 8288] to [-1, 1]
            state[1] = state[1] / 8288
        if len(state) > 2:
            # Normalize facing from [0; 360] to [-1, 1]
            state[2] = (state[2] - 180) / 180
        return state
