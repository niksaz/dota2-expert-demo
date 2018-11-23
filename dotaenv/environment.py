import time

from tensorforce.environments import Environment

import dotaenv.bot_server as server
import dotaenv.dota_runner as runner
from dotaenv.codes import STATE_DIM, ACTIONS_TOTAL


class DotaEnvironment(Environment):

    def __init__(self):
        self.observation_space = (STATE_DIM,)
        self.action_space = (1,)
        self.terminal = False
        server.run_app()
        runner.prepare_dota_client()
        runner.start_game()

    def reset(self):
        runner.restart_game()
        return server.get_observation()[0]

    def execute(self, action):
        state, reward, terminal = server.step(action=action)
        self.terminal = terminal
        return state, reward, terminal

    def close(self):
        runner.close_game()

    def __str__(self):
        return 'Dota 2 Environment'

    @property
    def states(self):
        return dict(type='float', shape=self.observation_space)

    @property
    def actions(self):
        return dict(type='int', num_actions=ACTIONS_TOTAL)
