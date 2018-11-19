Reward = {}

local bot = GetBot()

local this_player_id = bot:GetPlayerID()

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
    return bot:GetHealth()
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
    return GetHeroKills(this_player_id)
end

function get_my_deaths()
    return GetHeroDeaths(this_player_id)
end

function get_distance_to_tower_punishment()
    return GetUnitToUnitDistance(bot, ally_tower) + GetUnitToUnitDistance(bot, enemy_tower)
end

function get_last_hits()
    return bot:GetLastHits()
end

function recently_damaged_enemy()
    local enemy = get_enemy()
    local result = 0
    if enemy ~= nil and enemy:WasRecentlyDamagedByAnyHero(5.0) then
        result = 1
    end

    return result
end

function is_near_ally_tower()
    local ally_tower = GetTower(TEAM_RADIANT, TOWER_MID_1)
    if ally_tower == nil then
        return 0
    end
    print('pos', bot:GetLocation())
    print('tower', ally_tower:GetLocation())
    print('tow dst', GetUnitToUnitDistance(bot, ally_tower))
    if GetUnitToUnitDistance(bot, ally_tower) < 1000 then
        return 1
    else
        return 0
    end
end

local last_attack_time = bot:GetLastAttackTime()

function Reward.get_reward(wrong_action)
--    local my_health = get_my_health()
--    local my_kills = get_my_kills()
--    local enemy_health = get_enemy_health()
--    local enemy_tower_health = get_enemy_tower_health()
--    local ally_tower_health = get_ally_tower_health()
--    local hits = get_last_hits()

--    local reward = -(my_deaths - last_deaths) * 10000 -- deaths
--            + (my_kills - last_kills) * 1000000 -- kills
--            + max(enemy_tower_health - last_enemy_tower_health, 0) * 5 * is_near_enemy_tower() -- enemy tower
--            - max(ally_tower_health - last_ally_tower_health, 0)
--            + max(enemy_health - last_enemy_health, 0) * 100 * recently_damaged_enemy()
--            - max(my_health - last_my_health, 0) * 100
--            + max(hits - last_hits, 0) * 10000
--            - get_distance_to_tower_punishment() / 200
--            - wrong_action * 30

    local reward = 0

    if is_near_ally_tower() == 1 then
        reward = reward + 1
    end

    local attack_time = bot:GetLastAttackTime()
    if attack_time ~= nil and (last_attack_time == nil or attack_time >
            last_attack_time) then
        reward = reward + 5
    end
    last_attack_time = attack_time

    if wrong_action == 1 then
        reward = reward - 0.1
    end

    local deaths = get_my_deaths()
    reward = reward - (deaths - last_deaths) * 100
    last_deaths = deaths

--    last_enemy_tower_health = enemy_tower_health
--    last_ally_tower_health = ally_tower_health
--    last_kills = my_kills
--    last_hits = hits
--    last_my_health = my_health
--    last_enemy_health = enemy_health

    print('reward: ', reward)
    return reward
end

return Reward
