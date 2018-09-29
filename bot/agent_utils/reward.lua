Reward = {}

local this_bot = GetBot()

local enemy_tower = GetTower(TEAM_DIRE, TOWER_MID_1);
local ally_tower = GetTower(TEAM_RADIANT, TOWER_MID_1);
if GetTeam() == TEAM_DIRE then
    local temp = ally_tower
    ally_tower = enemy_tower
    enemy_tower = temp
end

local this_player_id = this_bot:GetPlayerID()

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
    return this_bot:GetHealth()
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
    return GetUnitToUnitDistance(this_bot, ally_tower) + GetUnitToUnitDistance(this_bot, enemy_tower)
end

function get_last_hits()
    return this_bot:GetLastHits()
end

function recently_damaged_enemy()
    local enemy = get_enemy()
    local result = 0
    if enemy ~= nil and enemy:WasRecentlyDamagedByAnyHero(5.0) then
        result = 1
    end

    return result
end

function is_near_enemy_tower()
    if GetUnitToUnitDistance(this_bot, enemy_tower) < 1000 then
        return 1
    else
        return 0
    end
end

function Reward.get_reward(wrong_action)
    local my_health = get_my_health()
    local my_deaths = get_my_deaths()
    local my_kills = get_my_kills()
    local enemy_health = get_enemy_health()
    local enemy_tower_health = get_enemy_tower_health()
    local ally_tower_health = get_ally_tower_health()
    local hits = get_last_hits()

    local reward = -(my_deaths - last_deaths) * 10000 -- deaths
            + (my_kills - last_kills) * 1000000 -- kills
            + max(enemy_tower_health - last_enemy_tower_health, 0) * 5 * is_near_enemy_tower() -- enemy tower
            - max(ally_tower_health - last_ally_tower_health, 0)
            + max(enemy_health - last_enemy_health, 0) * 100 * recently_damaged_enemy()
            - max(my_health - last_my_health, 0) * 100
            + max(hits - last_hits, 0) * 10000
            - get_distance_to_tower_punishment() / 200
            - wrong_action * 30

    last_enemy_tower_health = enemy_tower_health
    last_ally_tower_health = ally_tower_health
    last_deaths = my_deaths
    last_kills = my_kills
    last_hits = hits
    last_my_health = my_health
    last_enemy_health = enemy_health

    print('Reward: ', reward)
    return reward
end

local target = GetTower(TEAM_RADIANT, TOWER_MID_1);

function Reward.get_distance_to_tower()
    return GetUnitToUnitDistance(this_bot, target)
end

-- TODO
function Reward.tower_distance_reward()
    local reward = GetUnitToUnitDistance(this_bot, target)
    reward = math.exp(-0.002 * reward + 30)
    print('Reward: ', reward)
    return reward
end

return Reward;