-- Selects heroes for the bots in the lobby.

local Config = require(GetScriptDirectory() .. '/config')

function Think()
    print("is_in_training_mode=" .. tostring(Config.is_in_training_mode))
    if Config.is_in_training_mode then
        if GetTeam() == TEAM_RADIANT then
            print("selecting radiant")
            SelectHero(2, "npc_dota_hero_nevermore")
            SelectHero(3, "npc_dota_hero_sven")
            SelectHero(4, "npc_dota_hero_sven")
            SelectHero(5, "npc_dota_hero_sven")
            SelectHero(6, "npc_dota_hero_sven")
        elseif GetTeam() == TEAM_DIRE then
            print("selecting dire")
            SelectHero(7, "npc_dota_hero_lina")
            SelectHero(8, "npc_dota_hero_sven")
            SelectHero(9, "npc_dota_hero_sven")
            SelectHero(10, "npc_dota_hero_sven")
            SelectHero(11, "npc_dota_hero_sven")
        end
    else
        if GetTeam() == TEAM_RADIANT then
            print("selecting radiant");
            SelectHero(2, "npc_dota_hero_nevermore");
            SelectHero(3, "npc_dota_hero_sven");
            SelectHero(4, "npc_dota_hero_enigma")
            SelectHero(5, "npc_dota_hero_sven");
            SelectHero(6, "npc_dota_hero_sven");
        elseif (GetTeam() == TEAM_DIRE) then
            print("selecting dire");
            SelectHero(7, "npc_dota_hero_lina");
            SelectHero(8, "npc_dota_hero_sven");
            SelectHero(9, "npc_dota_hero_sven");
            SelectHero(10, "npc_dota_hero_sven");
            SelectHero(11, "npc_dota_hero_sven");
        end
    end
end

if (GetTeam() == TEAM_DIRE) then
    function UpdateLaneAssignments()
        return {
            [7] = LANE_MID,
            [8] = LANE_MID,
            [9] = LANE_MID,
            [10] = LANE_MID,
            [11] = LANE_MID,
        }
    end
end
