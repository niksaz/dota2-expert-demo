import gym
from gym import spaces
import logging.config
import numpy as np

import dotaenv.bot_server as server
import dotaenv.dota_runner as runner
from dotaenv.codes import STATE_DIM, ACTIONS_TOTAL


RESTART_AFTER_EPISODES = 10


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

        self.episodes_experienced = 0
        server.run_app()

    def step(self, action):
        state, reward, terminal, info = server.step(action=action)
        return state, reward, terminal, info

    def reset(self):
        server.reset()
        self.episodes_experienced += 1
        if self.episodes_experienced > RESTART_AFTER_EPISODES:
            self.episodes_experienced = 0
            self.close()
        runner.restart_game()
        observation, _, _, _ = server.get_observation()
        # Sometimes the Dota 2 client takes more time than planned and then
        # we need to reset it once again
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
