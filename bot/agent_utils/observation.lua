-- Observation generation module.

local Observation = {}

local MAX_ABS_X = 8288.0
local MAX_ABS_Y = 8288.0

local Resolver = require(GetScriptDirectory() .. '/agent_utils/resolver')
local Func = require(GetScriptDirectory() .. '/util/func')
local Config = require(GetScriptDirectory() .. '/config')

local agent = Config.is_in_training_mode and GetBot() or GetTeamMember(1)
local agent_player_id = agent:GetPlayerID()

local enemy_creeps = {}
local enemy_heroes = {}
local enemy_towers = {}

local NEARBY_RADIUS = 1500

function get_hero_info()
    local hero_info = {}

    local self_position = agent:GetLocation()
    -- Normalized coordinates of the hero (original range is [-8288; 8288])
    Func.extend_table(hero_info, {
        self_position[1] / MAX_ABS_X,
        self_position[2] / MAX_ABS_Y
    })
    -- Possibility of moving in the directions {0 -- allowed, 1 -- disallowed}
    for dir=0,(Resolver.total_dirs-1) do
        local dir_vector = Resolver.delta_vector_for_dir(dir)
        Func.extend_table(hero_info, {
            Resolver.can_move_by_delta(self_position, dir_vector) and 0 or 1
        })
    end
    -- Info about health
    Func.extend_table(hero_info, { agent:GetHealth() / agent:GetMaxHealth()})
    return hero_info
end

function get_enemy_info()
    local enemy_info = {}
    -- Info about nearby enemy creeps
    if #enemy_creeps > 0 then
        local creep = enemy_creeps[1]
        local creep_dst = GetUnitToUnitDistance(agent, creep) / NEARBY_RADIUS
        Func.extend_table(enemy_info, {0, creep_dst})
    else
        Func.extend_table(enemy_info, {1, 1})
    end
    -- Info about nearby enemy heroes
    if #enemy_heroes > 0 then
        local hero = enemy_heroes[1]
        local hero_dst = GetUnitToUnitDistance(agent, hero) / NEARBY_RADIUS
        Func.extend_table(enemy_info, {0, hero_dst})
    else
        Func.extend_table(enemy_info, {1, 1})
    end
    -- Info about nearby enemy towers
    if #enemy_towers > 0 then
        local tower = enemy_towers[1]
        local tower_dst = GetUnitToUnitDistance(agent, tower) / NEARBY_RADIUS
        Func.extend_table(enemy_info, {0, tower_dst})
    else
        Func.extend_table(enemy_info, {1, 1})
    end
    return enemy_info
end

function Observation.update_info_about_environment()
    enemy_creeps = agent:GetNearbyCreeps(NEARBY_RADIUS, true)
    enemy_heroes = agent:GetNearbyHeroes(NEARBY_RADIUS, true, BOT_MODE_NONE)
    enemy_towers = agent:GetNearbyTowers(NEARBY_RADIUS, true)
end

-- Get all observations.
function Observation.get_observation()
    local observation = {
        ['hero_info'] = get_hero_info(),
        ['enemy_info'] = get_enemy_info(),
    }
    print('observation:', Func.dump(observation))
    return observation
end

-- Get the info about the agent's actions.
function Observation.get_action_info()
    local future_loc = agent:GetExtrapolatedLocation(0.1) -- in seconds
    local target = agent:GetAttackTarget()
    local attacking_creep = -1
    if enemy_creeps ~= nil and target ~= nil then
        attacking_creep = Func.in_table(enemy_creeps, target)
    end
    local attacking_hero = -1
    if enemy_heroes ~= nil and target ~= nil then
        attacking_hero = Func.in_table(enemy_heroes, target)
    end
    local attacking_tower = -1
    if enemy_towers ~= nil and target ~= nil then
        attacking_tower = Func.in_table(enemy_towers, target)
    end
    local action_info = {
        future_loc[1] / MAX_ABS_X,
        future_loc[2] / MAX_ABS_Y,
        attacking_creep,
        attacking_hero,
        attacking_tower
    }
    return action_info
end

function Observation.is_done()
    local _end = false

    if GetGameState() == GAME_STATE_POST_GAME or
            GetHeroKills(agent_player_id) > 0 or
            GetHeroDeaths(agent_player_id) > 0 or
            DotaTime() > (Config.is_in_training_mode and 600 or 120) then
        _end = true
        print('Bot: the game has ended.')
    end

    return _end
end

return Observation
