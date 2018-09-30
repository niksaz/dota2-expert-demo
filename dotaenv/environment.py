# /usr/bin/env python3
import time

from tensorforce.environments import Environment

import dotaenv.bot_server as server
from dotaenv.dota_runner import start_game, set_timescale, launch_dota, \
    restart_game


class DotaEnvironment(Environment):

    def __init__(self):
        self.action_space = (21,)
        self.observation_space = (172,)
        self.terminal = False
        server.run_app()
        launch_dota()
        set_timescale()
        start_game()

    def reset(self):
        if self.terminal:
            restart_game()
            self.terminal = False
            time.sleep(5)
        return server.get_observation()[0]

    def execute(self, action):
        state, reward, terminal = server.step(action=action)
        self.terminal = terminal
        return state, terminal, reward

    @property
    def states(self):
        return dict(type='float', shape=(172,))

    @property
    def actions(self):
        return dict(
            action_type=dict(type='int', num_actions=5),
            move_vector=dict(type='int', num_actions=16),
            creep_index=dict(type='int', num_actions=10),
            ability_index=dict(type='int', num_actions=4)
        )
