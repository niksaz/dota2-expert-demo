-- Action execution module.

local Action = {}

local Resolver = require(GetScriptDirectory() .. '/agent_utils/resolver')
local Config = require(GetScriptDirectory() .. 'config')
local agent = Config.agent

local NEARBY_RADIUS = 1500
local ACTION_MOVE = 0
local ACTION_ATTACK_HERO = 1
local ACTION_ATTACK_CREEP = 2
local ACTION_USE_ABILITY = 3
local ACTION_ATTACK_TOWER = 4
local ACTION_MOVE_DISCRETE = 5
local ACTION_DO_NOTHING = 6

local wrong_action = 0

local ABILITY = {
    agent:GetAbilityByName('nevermore_shadowraze1'),
    agent:GetAbilityByName('nevermore_shadowraze2'),
    agent:GetAbilityByName('nevermore_shadowraze3'),
    agent:GetAbilityByName('nevermore_requiem')
}

--- Move by the delta vector.
-- @param delta_vector
--
function move_delta(delta_vector)
    print('MOVE BY DELTA', delta_vector[1], delta_vector[2])
    local position = agent:GetLocation()
    if Resolver.can_move_by_delta(position, delta_vector) then
        agent:Action_MoveDirectly(position + delta_vector)
    else
        wrong_action = 1
    end
end

--- Move towards the direction.
-- @param dir code
--
function move_discrete(dir)
    print('MOVE BY DIR', dir)
    local delta_vector = Resolver.delta_vector_for_dir(dir)
    move_delta(delta_vector)
end

--- Use ability.
-- @param ability_idx index of ability in 'ABILITY' table.
--
function use_ability(ability_idx)
    print('USE ABILITY', ability_idx)
    local ability = ABILITY[ability_idx]
    if ability:IsFullyCastable() then
        agent:Action_UseAbility(ability)
    else
        wrong_action = 1
    end
end

--- Attack the closest enemy hero nearby.
function attack_hero()
    print('ATTACK HERO')
    local enemy_heroes_list = agent:GetNearbyHeroes(NEARBY_RADIUS, true, BOT_MODE_NONE)
    if #enemy_heroes_list > 0 then
        agent:Action_AttackUnit(enemy_heroes_list[1], false)
    else
        wrong_action = 1
    end
end

--- Attack enemy creep.
-- @param creep_idx index of creep in nearby creeps table.
--
function attack_creep(creep_idx)
    print('ATTACK CREEP', creep_idx)
    local enemy_creeps = agent:GetNearbyCreeps(NEARBY_RADIUS, true)
    if #enemy_creeps >= creep_idx then
        agent:Action_AttackUnit(enemy_creeps[creep_idx], false)
    else
        wrong_action = 1
    end
end

-- Attack nearby enemy tower.
function attack_tower()
    print('ATTACK TOWER')
    local towers = agent:GetNearbyTowers(NEARBY_RADIUS, true)
    if #towers > 0 then
        agent:Action_AttackUnit(towers[1], false)
    else
        wrong_action = 1
    end
end

-- Take all available abilities' upgrades.
function upgrade_abilities()
    for _, ability in ipairs(ABILITY) do
        if ability:CanAbilityBeUpgraded() then
            agent:ActionImmediate_LevelAbility(ability:GetName())
        end
    end
end

--- Execute given action.
-- @param action_info action info {'action': action id, 'params': action parameters}
--
function Action.execute_action(action_info)
    local action = action_info['action']
    local action_params = action_info['params']
    wrong_action = 0

    upgrade_abilities()

    if action == ACTION_MOVE then
        -- Consider params[1], params[2] as x, y of a delta vector
        move_delta(action_params)
    elseif action == ACTION_MOVE_DISCRETE then
        -- Move towards the angle in degrees
        move_discrete(action_params[1])
    elseif action == ACTION_USE_ABILITY then
        -- Consider params[1] as an ability index
        use_ability(action_params[1])
    elseif action == ACTION_ATTACK_HERO then
        -- Attacks the closest enemy hero.
        attack_hero()
    elseif action == ACTION_ATTACK_CREEP then
        -- Consider params[1] as an index in the nearby creeps table
        attack_creep(action_params[1])
    elseif action == ACTION_ATTACK_TOWER then
        -- Attacks the closest enemy tower.
        attack_tower()
    elseif action == ACTION_DO_NOTHING then
        -- do nothing
    end

    return wrong_action
end

return Action
