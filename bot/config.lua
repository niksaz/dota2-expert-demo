-- Stores the configuration of the bot.

local Config = {}

-- If it is false, then the bot is in observer mode.
Config.is_in_training_mode = true
Config.agent_to_act = GetBot()
Config.agent_to_observe = Config.is_in_training_mode and GetBot() or GetTeamMember(1)

return Config
