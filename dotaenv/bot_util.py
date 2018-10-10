#!/usr/bin/env python3

import numpy as np


def action_to_json(action):
    """
    Lua indexes starts from 1!

    local ACTION_MOVE = 0
    local ACTION_ATTACK_HERO = 1
    local ACTION_ATTACK_CREEP = 2
    local ACTION_USE_ABILITY = 3
    local ACTION_ATTACK_TOWER = 4
    local ACTION_MOVE_DISCRETE = 5
    local ACTION_DO_NOTHING = 6

    :param action: action numeric code
    :return: bot-compatible JSON action message
    """

    params = []
    if 0 <= action < 16:
        # move
        params.append(int(action))
        action = 5
    elif 16 <= action < 26:
        # attack creep
        params.append(int(action - 16) + 1)
        action = 2
    elif 26 <= action < 30:
        # use ability
        params.append(int(action - 26) + 1)
        action = 3
    elif action == 30:
        # attack hero
        action = 1
    elif action == 31:
        # attack tower
        action = 4
    elif action == 32:
        # do nothing
        action = 6

    action_response = {
        'action': action,
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
    else:
        observation = []
        reward = 0.
        done = True
    return observation[:3], reward, done  # TODO: Taking prefix slice.


def vectorize_observation(observation):
    result = []
    # bot:GetLocation[1]
    # bot:GetLocation[2]
    # bot:GetFacing()
    # bot:GetAttackDamage()
    # bot:GetLevel()
    # bot:GetHealth()
    # bot:GetMana()
    # is_castable_ability1
    # is_castable_ability2
    # is_castable_ability3
    # is_castable_ability4
    # 11 values
    result.extend(observation['self_info'])
    # enemy_position[1]
    # enemy_position[2]
    # enemy:GetAttackDamage()
    # enemy:GetLevel()
    # enemy:GetHealth()
    # enemy:GetMana()
    # enemy:GetFacing()
    # 7 values
    result.extend(observation['enemy_info'])

    # info about 10 nearby enemy creeps
    # creep:GetHealth()
    # creep:GetLocation[1]
    # creep:GetLocation[2]
    # 3x10=30 values
    creeps = observation['enemy_creeps_info']
    for i in range(10):
        if i < len(creeps):
            result.extend(creeps[i])
        else:
            result.extend([0, 0, 0])

    # info about 10 nearby allied creeps
    # creep:GetHealth()
    # creep:GetLocation[1]
    # creep:GetLocation[2]
    # 3x10=30 values
    creeps = observation['ally_creeps_info']
    for i in range(10):
        if i < len(creeps):
            result.extend(creeps[i])
        else:
            result.extend([0, 0, 0])

    # enemy_tower: GetHealth()
    # ally_tower: GetHealth()
    # 2 values
    result.extend(observation['tower_info'])
    # bot: TimeSinceDamagedByAnyHero()
    # bot: TimeSinceDamagedByCreep()
    # bot: TimeSinceDamagedByTower()
    # 3 values
    result.extend(observation['damage_info'])

    # 83 values
    return np.array(result)
