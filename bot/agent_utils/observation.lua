-- Observation generation module.

local Observation = {}

local Resolver = require(GetScriptDirectory() .. '/agent_utils/resolver')
local Func = require(GetScriptDirectory() .. '/util/func')
local Config = require(GetScriptDirectory() .. '/config')

local agent = Config.is_in_training_mode and GetBot() or GetTeamMember(1)
local agent_player_id = agent:GetPlayerID()

local NEARBY_RADIUS = 1500

function get_hero_info()
    local hero_info = {}

    local self_position = agent:GetLocation()
    -- Normalized coordinates of the hero (original range is [-8288; 8288])
    Func.extend_table(hero_info, {
        self_position[1] / 8288.0,
        self_position[2] / 8288.0
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
    local enemy_creeps = agent:GetNearbyCreeps(NEARBY_RADIUS, true)
    if #enemy_creeps > 0 then
        local creep = enemy_creeps[1]
        local creep_dst = GetUnitToUnitDistance(agent, creep) / NEARBY_RADIUS
        Func.extend_table(enemy_info, {0, creep_dst})
    else
        Func.extend_table(enemy_info, {1, 1})
    end
    -- Info about nearby enemy heroes
    local enemy_heroes = agent:GetNearbyHeroes(NEARBY_RADIUS, true, BOT_MODE_NONE)
    if #enemy_heroes > 0 then
        local hero = enemy_heroes[1]
        local hero_dst = GetUnitToUnitDistance(agent, hero) / NEARBY_RADIUS
        Func.extend_table(enemy_info, {0, hero_dst})
    else
        Func.extend_table(enemy_info, {1, 1})
    end
    -- Info about nearby enemy towers
    local enemy_towers = agent:GetNearbyTowers(NEARBY_RADIUS, true)
    if #enemy_towers > 0 then
        local tower = enemy_towers[1]
        local tower_dst = GetUnitToUnitDistance(agent, tower) / NEARBY_RADIUS
        Func.extend_table(enemy_info, {0, tower_dst})
    else
        Func.extend_table(enemy_info, {1, 1})
    end
    return enemy_info
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

function Observation.is_done()
    local _end = false

    if GetGameState() == GAME_STATE_POST_GAME or
            GetHeroKills(agent_player_id) > 0 or
            GetHeroDeaths(agent_player_id) > 0 or
            DotaTime() > (Config.is_in_training_mode and 360 or 120) then
        _end = true
        print('Bot: the game has ended.')
    end

    return _end
end

return Observation
