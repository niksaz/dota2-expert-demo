from enum import IntEnum
from threading import Condition, Thread
from flask import Flask
from flask import request
from flask import jsonify
from flask import abort
import logging

from dotaenv.bot_util import message_to_pairs, action_to_json

logger = logging.getLogger('dota2env.bot_server')

app = Flask(__name__)


def run_app(port=5000):
    """
    Run bot server application in separate thread.

    :param port: port to run on
    :return application thread
    """
    logger.debug('Starting bot server on port {port}.'.format(port=port))
    app_thread = Thread(target=lambda: app.run(port=port))
    app_thread.setDaemon(True)
    app_thread.start()
    return app_thread


class FsmState(IntEnum):
    IDLE = 0
    ACTION_RECEIVED = 1
    SEND_OBSERVATION = 2


changed_condition = Condition()
observation = None  # Guarded by changed_condition
current_action = None  # Guarded by changed_condition
is_reset = True  # Guarded by changed_condition


def reset():
    """
    Returns the server to the initial state and notifies all waiting for an action threads.
    """
    global observation, current_action, is_reset
    changed_condition.acquire()
    observation = None
    current_action = None
    is_reset = True
    changed_condition.notify_all()
    changed_condition.release()


def get_observation_pairs():
    """
    Gets an observation from the dota thread.

    :return: tuple (observation, reward, is_done)
    """
    global observation
    changed_condition.acquire()
    while observation is None:
        # wait for the dota thread to produce an observation
        timeout_satisfied = changed_condition.wait(timeout=30)
        if not timeout_satisfied:
            break

    result = observation
    observation = None
    changed_condition.notify_all()
    changed_condition.release()

    return message_to_pairs(result)


def step(action):
    """
    Executes the action and receives an observation from the bot.

    :return: tuple (observation, reward, is_done)
    """
    global current_action

    changed_condition.acquire()
    while current_action is not None:
        # wait for the dota thread to consume the action
        timeout_satisfied = changed_condition.wait(timeout=30)
        if not timeout_satisfied:
            break

    current_action = action_to_json(action)
    changed_condition.notify_all()
    changed_condition.release()

    return get_observation_pairs()


@app.route('/observation', methods=['POST'])
def process_observation():
    global observation, current_action, is_reset

    changed_condition.acquire()
    is_reset = False
    while observation is not None:
        # wait for the agent to consume the observation
        changed_condition.wait()
        if is_reset:
            changed_condition.release()
            abort(404)

    observation = request.get_json()['content']
    changed_condition.notify_all()

    while current_action is None:
        # wait for the agent to produce an action
        changed_condition.wait()
        if is_reset:
            changed_condition.release()
            abort(404)

    response = jsonify({'fsm_state': FsmState.ACTION_RECEIVED, 'action': current_action})
    current_action = None
    changed_condition.notify_all()
    changed_condition.release()

    return response
