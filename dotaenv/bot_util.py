#!/usr/bin/env python3

import numpy as np
from dotaenv.codes import MOVE_ACTIONS_TOTAL, ATTACK_CREEP, ATTACK_HERO, ATTACK_TOWER, STATE_PROJECT


def action_to_json(action_internal):
    """
    Lua indexes starts from 1!

    local ACTION_MOVE = 0
    local ACTION_ATTACK_HERO = 1
    local ACTION_ATTACK_CREEP = 2
    local ACTION_USE_ABILITY = 3
    local ACTION_ATTACK_TOWER = 4
    local ACTION_MOVE_DISCRETE = 5
    local ACTION_DO_NOTHING = 6

    :param action_internal: action numeric code
    :return: bot-compatible JSON action message
    """

    bot_action = 6
    params = []
    if 0 <= action_internal < MOVE_ACTIONS_TOTAL:
        # move
        bot_action = 5
        dir_code = int(action_internal)
        params.append(dir_code)
    elif action_internal == ATTACK_CREEP:
        # attack the nearest creep
        bot_action = 2
        params.append(1)
    elif action_internal == ATTACK_HERO:
        # attack the enemy hero
        bot_action = 1
    elif action_internal == ATTACK_TOWER:
        # attack the enemy tower
        bot_action = 4

    action_response = {
        'action': bot_action,
        'params': params
    }
    return action_response


def message_to_observation(observation_message):
    """
    Transform bot observation message to
    :param observation_message:
    :return:
    """
    if observation_message is not None:
        observation = vectorize_observation(observation_message['observation'])
        reward = observation_message['reward']
        done = observation_message['done']
        action_info = vectorize_action_info(observation_message['action_info'])
    else:
        observation = []
        reward = 0.
        done = True
        action_info = []
    return observation, reward, done, action_info


def vectorize_observation(observation):
    result = []
    result.extend(observation['hero_info'])
    result.extend(observation['enemy_info'])
    return np.array(result, dtype=np.float32)[STATE_PROJECT]


def vectorize_action_info(action_info):
    return np.array(action_info, dtype=np.float32)
