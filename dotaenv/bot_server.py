from enum import IntEnum
from threading import Event, Thread, RLock
from flask import Flask
from flask import request
from flask import jsonify
import logging

from dotaenv.bot_util import message_to_observation, action_to_json

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
    WHAT_NEXT = 0
    ACTION_RECEIVED = 1
    SEND_OBSERVATION = 2


lock = RLock()
observation_received = Event()
observation = None
current_action = None
current_fsm_state = FsmState.WHAT_NEXT


def step(action):
    """
    Execute action and receive observation from the bot.

    :return: tuple (observation, reward, is_done)
    """
    global current_fsm_state, current_action, observation

    lock.acquire()
    current_fsm_state = FsmState.ACTION_RECEIVED
    current_action = action_to_json(action)
    lock.release()

    observation_received.wait(timeout=20)
    result = observation
    observation = None
    observation_received.clear()

    return message_to_observation(result)


def get_observation():
    """
    Get observation from the bot.

    :return: tuple (observation, reward, is_done)
    """
    global current_fsm_state, observation

    lock.acquire()
    current_fsm_state = FsmState.SEND_OBSERVATION
    lock.release()

    observation_received.wait()
    result = observation
    observation = None
    observation_received.clear()

    return message_to_observation(result)


def bot_response():
    global current_action, current_fsm_state
    lock.acquire()
    if current_action is None:
        current_fsm_state = FsmState.SEND_OBSERVATION
    else: 
        current_fsm_state = FsmState.ACTION_RECEIVED
    response = jsonify({'fsm_state': current_fsm_state, 'action': current_action})
    current_action = None
    lock.release()
    return response


@app.route('/observation', methods=['POST'])
def process_observation():
    global observation, current_fsm_state
    observation = request.get_json()['content']
    observation_received.set()
    response = bot_response()
    return response
