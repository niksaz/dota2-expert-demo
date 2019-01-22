require(GetScriptDirectory() .. '/util/json')

local Observation = require(GetScriptDirectory() .. '/agent_utils/observation')
local Reward = require(GetScriptDirectory() .. '/agent_utils/reward')
local Action = require(GetScriptDirectory() .. '/agent_utils/action')

-- How many frames should pass before a new observation is sent
local MIN_FRAMES_BETWEEN = 1

local frame_count = 0
local total_reward = 0
local current_action = 0
local action_to_do_next

-- Bot communication automaton.
local IDLE = 0
local ACTION_RECEIVED = 1
local SEND_OBSERVATION = 2
local fsm_state = SEND_OBSERVATION

local wrong_action = 0

--- Executes received action.
-- @param action_info bot action
--
function execute_action(action)
    wrong_action = Action.execute_action(action)
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
                    action_to_do_next = response['action']
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

--- Send JSON with the message.
--
function send_observation_message(msg)
    send_message(create_message(msg, 'observation'), '/observation', nil)
end

function Think()
    total_reward = total_reward + Reward.get_reward(wrong_action)
    frame_count = frame_count + 1
    -- Decide on what to do next based on the state
    if fsm_state == SEND_OBSERVATION then
        fsm_state = IDLE
        local observation = Observation.get_observation(current_action)
        local done = Observation.is_done()
        message = {
            current_action, {
                ['observation'] = observation,
                ['reward'] = total_reward,
                ['done'] = done,
            }
        }
        send_observation_message({message})
        print('FRAME COUNT', frame_count)
        frame_count = 0
        if done then
            DebugPause()
        end
    elseif fsm_state == ACTION_RECEIVED and frame_count + 1 >= MIN_FRAMES_BETWEEN then
        fsm_state = SEND_OBSERVATION
        current_action = action_to_do_next
    elseif fsm_state == IDLE then
        -- Do nothing
    end
    -- Act
    execute_action(current_action)
end
