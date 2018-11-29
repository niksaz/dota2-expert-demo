from tensorforce.environments import Environment

import dotaenv.bot_server as server
import dotaenv.dota_runner as runner
from dotaenv.codes import STATE_DIM, ACTIONS_TOTAL


RESTART_AFTER_EPISODES = 100


class DotaEnvironment(Environment):

    def __init__(self):
        self.observation_space = (STATE_DIM,)
        self.action_space = (1,)
        self.episodes_experienced = 0
        server.run_app()

    def reset(self):
        self.episodes_experienced += 1
        if self.episodes_experienced > RESTART_AFTER_EPISODES:
            self.episodes_experienced = 0
            self.close()
        runner.restart_game()
        return server.get_observation()[0]

    def execute(self, action):
        state, reward, terminal = server.step(action=action)
        return state, reward, terminal

    def close(self):
        runner.close_dota_client()
        runner.close_steam_client()

    def __str__(self):
        return 'Dota 2 Environment'

    @property
    def states(self):
        return dict(type='float', shape=self.observation_space)

    @property
    def actions(self):
        return dict(type='int', num_actions=ACTIONS_TOTAL)
