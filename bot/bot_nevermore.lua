require(GetScriptDirectory() .. '/util/json')

local Observation = require(GetScriptDirectory() .. '/agent_utils/observation')
local Reward = require(GetScriptDirectory() .. '/agent_utils/reward')
local Action = require(GetScriptDirectory() .. '/agent_utils/action')

local current_action
local state_num = 0

-- Bot communication automaton.
local IDLE = 0
local ACTION_RECEIVED = 1
local SEND_OBSERVATION = 2
local fsm_state = SEND_OBSERVATION

local wrong_action = 0

--- Executes received action.
-- @param action_info bot action
--
function execute_action(action_info)
    print("Execute order.", action_info)
    wrong_action = Action.execute_action(action_info)
end

--- Create JSON message from table 'message' of type 'type'.
-- @param message table containing message
-- @param type type of message e.g. 'what_next' or 'observation'
-- @return JSON encoded {'type': type, 'content': message}
--
function create_message(message, type)
    local msg = {
        ['type'] = type,
        ['content'] = message
    }

    local encode_msg = Json.Encode(msg)
    return encode_msg
end

--- Send JSON message to bot server.
-- @param json_message message to send
-- @param route route ('/what_next' or '/observation')
-- @param callback called after response is received
--
function send_message(json_message, route, callback)
    local req = CreateHTTPRequest(':5000' .. route)
    req:SetHTTPRequestRawPostBody('application/json', json_message)
    req:Send(function(result)
        for k, v in pairs(result) do
            if k == 'Body' then
                if v ~= '' then
                    local response = Json.Decode(v)
                    if callback ~= nil then
                        callback(response)
                    end
                    current_action = response['action']
                    fsm_state = response['fsm_state']
                else
                    fsm_state = WHAT_NEXT
                end
            end
        end
    end)
end

--- Ask what to do next.
--
function send_what_next_message()
    local message = create_message('', 'what_next')
    send_message(message, '/what_next', nil)
end

--- Send JSON with current state info.
--
function send_observation_message()
    local msg = {
        ['observation'] = Observation.get_observation(),
        ['reward'] = Reward.get_reward(wrong_action),
        ['done'] = Observation.is_done(),
        ['state_num'] = state_num
    }

    send_message(create_message(msg, 'observation'), '/observation', nil)
    state_num = state_num + 1
end

local last_time_sent = GameTime()

function Think()
    --print(DotaTime())
    if fsm_state == SEND_OBSERVATION then
        print('Sending')
        fsm_state = IDLE
        send_observation_message()
        last_time_sent = GameTime()
    elseif fsm_state == ACTION_RECEIVED then
        fsm_state = SEND_OBSERVATION
        execute_action(current_action)
    elseif fsm_state == IDLE then
        -- Do nothing
    end
end
