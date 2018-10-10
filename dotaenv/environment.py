# /usr/bin/env python3
import time
import pyautogui as gui

from tensorforce.environments import Environment

import dotaenv.bot_server as server
from dotaenv.dota_runner import start_game, set_timescale, launch_dota, \
    restart_game


class DotaEnvironment(Environment):

    def __init__(self):
        self.action_space = (16,)
        self.observation_space = (83,)
        self.terminal = False
        server.run_app()
        launch_dota()
        set_timescale()
        start_game()

    def reset(self):
        if self.terminal:
            restart_game()
            self.terminal = False
            time.sleep(10)
            gui.press('esc', pause=1)
            gui.press('esc', pause=1)
            gui.press('esc', pause=1)
            gui.press('esc', pause=1)
            gui.press('esc', pause=1)
        return server.get_observation()[0]

    def execute(self, action):
        state, reward, terminal = server.step(action=action)
        self.terminal = terminal
        return state, reward, terminal

    @property
    def states(self):
        return dict(type='float', shape=self.observation_space)

    @property
    def actions(self):
        return dict(type='int', num_actions=self.action_space[0])
