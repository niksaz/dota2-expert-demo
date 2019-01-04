# Author: Mikita Sazanovich

STATE_PROJECT = range(17)
STATE_DIM = len(STATE_PROJECT)
SHAPER_STATE_PROJECT = range(2)
SHAPER_STATE_DIM = len(SHAPER_STATE_PROJECT)

# First actions are movement actions towards (360 / total * action_number)°
MOVE_ACTIONS_TOTAL = 8
# Action to attack the nearest creep
ATTACK_CREEP = MOVE_ACTIONS_TOTAL
# Action to attack the enemy hero
ATTACK_HERO = ATTACK_CREEP + 1
# Action to attack the enemy middle tower
ATTACK_TOWER = ATTACK_HERO + 1

ACTIONS_TOTAL = MOVE_ACTIONS_TOTAL + 3
