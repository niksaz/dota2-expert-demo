-- Action execution module.

local Action = {}

local Resolver = require(GetScriptDirectory() .. '/agent_utils/resolver')
local agent = GetBot()

Action.TOTAL_ACTIONS = 11

local NEARBY_RADIUS = 1600

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
    local delta_vector = Resolver.delta_vector_for_dir(dir)
    move_delta(delta_vector)
end

--- Use ability.
-- @param ability_idx index of ability in 'ABILITY' table.
--
function use_ability(ability_idx)
    local ability = ABILITY[ability_idx]
    if ability:IsFullyCastable() then
        agent:Action_UseAbility(ability)
    else
        wrong_action = 1
    end
end

--- Attack the closest enemy hero nearby.
function attack_hero()
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
    local enemy_creeps = agent:GetNearbyCreeps(NEARBY_RADIUS, true)
    if #enemy_creeps >= creep_idx then
        agent:Action_AttackUnit(enemy_creeps[creep_idx], false)
    else
        wrong_action = 1
    end
end

-- Attack nearby enemy tower.
function attack_tower()
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
function Action.execute_action(action)
    wrong_action = 0

    upgrade_abilities()

    if 0 <= action and action < Resolver.total_dirs then
        -- Move towards the angle in degrees
        move_discrete(action)
    elseif action == Resolver.total_dirs + 0 then
        -- Attack the closest creep.
        attack_creep(1)
    elseif action == Resolver.total_dirs + 1 then
        -- Attack the closest enemy hero.
        attack_hero()
    elseif action == Resolver.total_dirs + 2 then
        -- Attack the closest enemy tower.
        attack_tower()
    else
        -- Do nothing.
    end
    return wrong_action
end

return Action
