-- Utility functions.

local Func = {}

function Func.dump(o)
    if type(o) == 'table' then
        local s = '{'
        for k,v in pairs(o) do
            if type(k) ~= 'number' then k = '"'..k..'"' end
            s = s .. '[' .. k .. '] = ' .. Func.dump(v) .. ','
        end
        return s .. '} '
    else
        return tostring(o)
    end
end

function Func.extend_table(t1, t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

function Func.in_table(list, item)
    for key, value in pairs(list) do
        if value == item then return key end
    end
    return -1
end

return Func
