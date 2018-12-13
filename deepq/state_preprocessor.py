# Author: Mikita Sazanovich


class StatePreprocessor:

    @staticmethod
    def process(state):
        # Currently we do not preprocess the state returned from the Dota 2
        return state

    @staticmethod
    def _process_coordinate(coord):
        # Normalize a coordinate from [-8288; 8288] to [-1, 1]
        return coord / 8288

    @staticmethod
    def _process_facing(angle):
        # Normalize facing from [0; 360] to [-1, 1]
        return (angle - 180) / 180
