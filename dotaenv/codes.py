# Author: Mikita Sazanovich

STATE_PROJECT = [0, 1, 2]  # [11, 12, 19, 20]
STATE_DIM = len(STATE_PROJECT)

# First actions are movement actions towards (360 / total * action_number)°
MOVE_ACTIONS_TOTAL = 8
# Action to attack the nearest creep
ATTACK_CREEP = MOVE_ACTIONS_TOTAL
# Action to attack the enemy hero
ATTACK_HERO = ATTACK_CREEP + 1
# Action to attack the enemy middle tower
ATTACK_TOWER = ATTACK_HERO + 1

ACTIONS_TOTAL = ATTACK_TOWER + 1
