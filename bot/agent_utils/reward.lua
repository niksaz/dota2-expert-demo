-- Reward distribution module.

local Reward = {}

local agent = GetBot()

local agent_player_id = agent:GetPlayerID()

local last_enemy_tower_health = 1300
local last_ally_tower_health = 1300

local last_enemy_health = 1000
local last_my_health = 1000

local last_kills = 0
local last_deaths = 0
local last_hits = 0

function max(x, y)
    if x >= y then
        return x
    else
        return y
    end
end

function get_enemy()
    local enemy_table = GetUnitList(UNIT_LIST_ENEMY_HEROES)
    local enemy
    if #enemy_table > 0 then
        enemy = enemy_table[1]
    end

    return enemy
end

function get_my_health()
    return agent:GetHealth()
end

function get_enemy_health()
    local enemy = get_enemy()
    local enemy_health
    if enemy ~= nil then
        enemy_health = enemy:GetHealth()
    else
        enemy_health = last_enemy_health
    end
    return enemy_health
end

function get_enemy_max_health()
    local enemy = get_enemy()
    local enemy_max_health
    if enemy ~= nil then
        enemy_max_health = enemy:GetMaxHealth()
    else
        enemy_max_health = last_enemy_health
    end
    return enemy_max_health
end

function get_enemy_tower_health()
    return enemy_tower:GetHealth()
end

function get_ally_tower_health()
    return ally_tower:GetHealth()
end

function get_my_kills()
    return GetHeroKills(agent_player_id)
end

function get_my_deaths()
    return GetHeroDeaths(agent_player_id)
end

function get_distance_to_tower_punishment()
    return GetUnitToUnitDistance(agent, ally_tower) + GetUnitToUnitDistance(agent, enemy_tower)
end

function get_last_hits()
    return agent:GetLastHits()
end

function recently_damaged_enemy()
    local enemy = get_enemy()
    local result = 0
    if enemy ~= nil and enemy:WasRecentlyDamagedByAnyHero(5.0) then
        result = 1
    end

    return result
end

function Reward.is_near_ally_tower()
    local ally_tower = GetTower(TEAM_RADIANT, TOWER_MID_1)
    if ally_tower == nil then
        return 0
    end
    if GetUnitToUnitDistance(agent, ally_tower) < 500 then
        return 1
    else
        return 0
    end
end

local last_attack_time = agent:GetLastAttackTime()
local last_attack_target_hero = false

function Reward.get_reward(wrong_action)
    local reward = 0

    local attack_time = agent:GetLastAttackTime()
    if attack_time ~= nil and (last_attack_time == nil or attack_time > last_attack_time) and last_attack_target_hero then
        reward = reward + 1
    end
    last_attack_time = attack_time

    local attack_target = agent:GetAttackTarget()
    if attack_target ~= nil and attack_target == get_enemy() then
        last_attack_target_hero = true
    else
        last_attack_target_hero = false
    end

    return reward
end

return Reward
