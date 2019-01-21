#!/usr/bin/env python3

import numpy as np
from dotaenv.codes import STATE_PROJECT


def action_to_json(action_internal):
    action_response = int(action_internal)
    return action_response


def message_to_pairs(messages):
    pairs = []
    if messages is not None:
        for action, observation_message in messages:
            observation = vectorize_observation(observation_message['observation'])
            reward = observation_message['reward']
            done = observation_message['done']
            pairs.append((action, (observation, reward, done, [])))
            if done:
                break
    else:
        pairs.append((0, ([], 0., True, [])))
    return pairs


def vectorize_observation(observation):
    result = []
    result.append(observation['action_info'])
    result.extend(observation['hero_info'])
    result.extend(observation['enemy_info'])
    return np.array(result, dtype=np.float32)[STATE_PROJECT]
