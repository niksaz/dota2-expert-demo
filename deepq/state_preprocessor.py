# Author: Mikita Sazanovich


class StatePreprocessor:

    @staticmethod
    def process(state):
        if len(state) > 0:
            # Normalize x position from [-10000; 10000] to [0, 1]
            state[0] = (state[0] - (-10000)) / 20000
        if len(state) > 1:
            # Normalize y position from [-10000; 10000] to [0, 1]
            state[1] = (state[1] - (-10000)) / 20000
        if len(state) > 2:
            # Normalize facing from [0; 360] to [0, 1]
            state[2] = state[2] / 360
        return state
