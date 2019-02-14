import gym
from gym import spaces
import logging.config
import numpy as np

import dotaenv.bot_server as server
import dotaenv.dota_runner as runner
from dotaenv.codes import STATE_DIM, ACTIONS_TOTAL


class DotaEnvironment(gym.Env):

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("DotaEnvironment-{}".format(self.__version__))

        self.action_space = spaces.Discrete(ACTIONS_TOTAL)

        low = np.zeros(STATE_DIM, dtype=np.float32)
        low[0] = -1.0  # For x coordinate
        low[1] = -1.0  # For y coordinate
        high = np.ones(STATE_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        server.run_app()

    def step(self, action):
        return server.step(action=action)

    def reset(self):
        server.reset()
        runner.restart_game()
        observation, _, _, _ = server.get_observation_pairs()[-1][1]  # Second element of the last pair
        # Check the validity of the result
        return observation if len(observation) != 0 else self.reset()

    def render(self, mode='human'):
        # It is rendered in the Dota 2 client
        return

    def close(self):
        runner.close_dota_client()
        runner.close_steam_client()

    def seed(self, seed=None):
        # Can not seed DotaEnvironment as it communicates with the Dota 2 client
        return
