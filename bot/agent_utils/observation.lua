-- Observation module
local Observation = {}

local bot = GetBot()

local ability1 = bot:GetAbilityByName('nevermore_shadowraze1')
local ability2 = bot:GetAbilityByName('nevermore_shadowraze2')
local ability3 = bot:GetAbilityByName('nevermore_shadowraze3')
local ability4 = bot:GetAbilityByName('nevermore_requiem')

local creep_zero_padding = { 0, 0, 0 }

-- Obtain team info.
local function get_team()
    if (GetTeam() == TEAM_RADIANT) then
        return 1
    else
        return -1
    end
end

-- Obtain damage info.
function get_damage_info()
    local damage_info = {
        bot:TimeSinceDamagedByAnyHero(),
        bot:TimeSinceDamagedByCreep(),
        bot:TimeSinceDamagedByTower(),
    }
    return damage_info
end

-- Obtain towers info.
function get_towers_info()
    local enemy_tower = GetTower(TEAM_DIRE, TOWER_MID_1);
    local ally_tower = GetTower(TEAM_RADIANT, TOWER_MID_1);
    if get_team() == -1 then
        local temp = ally_tower
        ally_tower = enemy_tower
        enemy_tower = temp
    end

    return {
        enemy_tower:GetHealth(),
        ally_tower:GetHealth()
    }
end

--- Obtain bot's info (specified for Nevermore).
--
function get_self_info()
    local ability1_dmg = 0
    if ability1:IsFullyCastable() then
        ability1_dmg = 1
    end

    local ability2_dmg = 0
    if ability2:IsFullyCastable() then
        ability2_dmg = 1
    end

    local ability3_dmg = 0
    if ability3:IsFullyCastable() then
        ability3_dmg = 1
    end

    local ability4_dmg = 0
    if ability4:IsFullyCastable() then
        ability4_dmg = 1
    end

    -- Bot's atk, hp, mana, abilities, position x, position y
    local self_position = bot:GetLocation()
    local self_info = {
        self_position[1],
        self_position[2],
        bot:GetAttackDamage(),
        bot:GetLevel(),
        bot:GetHealth(),
        bot:GetMana(),
        bot:GetFacing(),
        ability1_dmg,
        ability2_dmg,
        ability3_dmg,
        ability4_dmg,
    }

    return self_info
end

-- Obtain enemy hero info.
function get_enemy_info()
    local enemy_table = GetUnitList(UNIT_LIST_ENEMY_HEROES)
    local enemy
    if enemy_table ~= nil then
        enemy = enemy_table[1]
    end

    local enemy_hero_input = { 0, 0, 0, 0, 0, 0, 0 }
    if (enemy ~= nil) then
        local enemy_position = enemy:GetLocation()
        enemy_hero_input = {
            enemy_position[1],
            enemy_position[2],
            enemy:GetAttackDamage(),
            enemy:GetLevel(),
            enemy:GetHealth(),
            enemy:GetMana(),
            enemy:GetFacing(),
        }
    end

    return enemy_hero_input
end

-- Obtain creeps info.
function get_creeps_info(creeps)
    local creeps_info = {}

    if (creeps == nil) then
        table.insert(creeps_info, creep_zero_padding)
        return creeps_info
    end

    for creep_key, creep in pairs(creeps)
    do
        local position = creep:GetLocation()
        table.insert(creeps_info, {
            creep:GetHealth(),
            position[1],
            position[2]
        })
    end

    -- if creeps_info is empty:
    if #creeps_info == 0 then
        table.insert(creeps_info, creep_zero_padding)
    end

    return creeps_info
end

-- Get whole observation.
function Observation.get_observation()
    local enemy_creeps = get_creeps_info(bot:GetNearbyCreeps(1500, true))
    local ally_creeps = get_creeps_info(bot:GetNearbyCreeps(1500, false))

    local observation = {
        ['self_info'] = get_self_info(),
        ['enemy_info'] = get_enemy_info(),
        ['enemy_creeps_info'] = enemy_creeps,
        ['ally_creeps_info'] = ally_creeps,
        ['tower_info'] = get_towers_info(),
        ['damage_info'] = get_damage_info()
    }

    return observation
end

return Observation;
