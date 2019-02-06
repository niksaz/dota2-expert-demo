-- Verifies the validity of actions.

local Resolver = {}
Resolver.total_dirs = 8
Resolver.move_radius = 100

function verify_dir(dir)
    assert(0 <= dir and dir < Resolver.total_dirs,
           "Incorrect direction for a move: " .. dir)
end

function Resolver.delta_vector_for_dir(dir)
    verify_dir(dir)

    local theta = dir * (360 / Resolver.total_dirs)
    local theta_rad = theta * (math.pi / 180)
    local cos_theta = math.cos(theta_rad)
    local sin_theta = math.sin(theta_rad)
    return Vector(
        cos_theta * Resolver.move_radius,
        sin_theta * Resolver.move_radius,
        0)
end

function Resolver.can_move_by_delta(pos, delta)
    local new_pos = pos + delta

    if not IsLocationPassable(new_pos) then
        return false
    end

    local diff = new_pos[1] - new_pos[2]
    local sq = diff * diff
    if sq > 1500000 then
        return false
    end

    return true
end

return Resolver
