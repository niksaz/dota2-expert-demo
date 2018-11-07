from tensorforce.environments import Environment

import dotaenv.bot_server as server
import dotaenv.dota_runner as runner
from dotaenv.codes import STATE_DIM, MOVES_TOTAL


class DotaEnvironment(Environment):

    def __init__(self):
        self.action_space = (MOVES_TOTAL,)
        self.observation_space = (STATE_DIM,)
        self.terminal = False
        self.restarts = 0
        server.run_app()
        runner.make_sure_dota_is_launched()
        runner.set_timescale()
        runner.start_game()

    def reset(self):
        runner._bring_into_focus()
        if self.restarts > 10:
            self.restarts = 0
            runner.close_game()
            runner.make_sure_dota_is_launched()
            runner.set_timescale()
            runner.start_game()
        else:
            self.restarts += 1
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
        return dict(type='int', num_actions=self.action_space[0])
