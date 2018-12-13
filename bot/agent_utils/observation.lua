-- Observation module

local Observation = {}

local Resolver = require(GetScriptDirectory() .. '/agent_utils/resolver')

local bot = GetBot()
local bot_player_id = bot:GetPlayerID()

local NEARBY_RADIUS = 1500
--local ability1 = bot:GetAbilityByName('nevermore_shadowraze1')
--local ability2 = bot:GetAbilityByName('nevermore_shadowraze2')
--local ability3 = bot:GetAbilityByName('nevermore_shadowraze3')
--local ability4 = bot:GetAbilityByName('nevermore_requiem')

function get_hero_info()
    local hero_info = {}

    local self_position = bot:GetLocation()
    -- Normalized coordinates of the hero (original range is [-8288; 8288])
    table.insert(hero_info, {
        self_position[1] / 8288.0,
        self_position[2] / 8288.0
    })
    -- Possibility of moving in the directions {0 -- allowed, 1 -- disallowed}
    for dir=0,(Resolver.total_dirs-1) do
        local dir_vector = Resolver.delta_vector_for_dir(dir)
        table.insert(hero_info, {
            Resolver.can_move_by_delta(self_position, dir_vector) and 0 or 1
        })
    end
    -- Info about health
    table.insert(hero_info, { bot:GetHealth() / bot:GetMaxHealth()})
    return hero_info
end

function get_enemy_info()
    local enemy_info = {}
    -- Info about nearby enemy creeps
    local enemy_creeps = get_creeps_info(bot:GetNearbyCreeps(NEARBY_RADIUS, true))
    if #enemy_creeps > 0 then
        local creep = creeps[1]
        local creep_dst = GetUnitToUnitDistance(bot, creep) / NEARBY_RADIUS
        table.insert(enemy_info, {0, creep_dst})
    else
        table.insert(enemy_info, {1, 1})
    end
    -- Info about nearby enemy heroes
    local enemy_heroes = bot:GetNearbyHeroes(NEARBY_RADIUS, true, BOT_MODE_NONE)
    if #enemy_heroes > 0 then
        local hero = enemy_heroes[1]
        local hero_dst = GetUnitToUnitDistance(bot, hero) / NEARBY_RADIUS
        table.insert(enemy_info, {0, hero_dst})
    else
        table.insert(enemy_info, {1, 1})
    end
    -- Info about nearby enemy towers
    local enemy_towers = bot:GetNearbyTowers(NEARBY_RADIUS, true)
    if #enemy_towers > 0 then
        local tower = enemy_towers[1]
        local tower_dst = GetUnitToUnitDistance(bot, tower) / NEARBY_RADIUS
        table.insert(enemy_info, {0, tower_dst})
    else
        table.insert(enemy_info, {1, 1})
    end
    return enemy_info
end

-- Get all observations.
function Observation.get_observation()
    local observation = {
        ['hero_info'] = get_hero_info(),
        ['enemy_info'] = get_enemy_info(),
    }
    print('observation:', dump(observation))
    return observation
end

function dump(o)
    if type(o) == 'table' then
        local s = '{'
        for k,v in pairs(o) do
            if type(k) ~= 'number' then k = '"'..k..'"' end
            s = s .. '[' .. k .. '] = ' .. dump(v) .. ','
        end
        return s .. '} '
    else
        return tostring(o)
    end
end

function Observation.is_done()
    local _end = false

    if GetGameState() == GAME_STATE_POST_GAME or
            GetHeroKills(bot_player_id) > 0 or
            GetHeroDeaths(bot_player_id) > 0 or
            DotaTime() > 360 then
        _end = true
        print('Bot: the game has ended.')
    end

    return _end
end

return Observation
