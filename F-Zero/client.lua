Buttons = {
	"A",
	"B",
	"X",
	"Y",
	"Up",
	"Down",
	"Left",
	"Right",
	"Start",
	"L",
	"R"
}

-- emu.limitframerate(true)
local checkpoint = 0x001162
local power = 0x0000C9
local reversed = 0x000B01
local game_over = 0x000048
local speed = 0x000B20

function setInput(actionstring)
	local input = {}
	for i = 1, #actionstring do
		local c = actionstring:sub(i,i)
		if c == '0' then
			input[Buttons[i]] = false
		else
			input[Buttons[i]] = true
		end
	end
	return input
end

-- main loop
local HOST = 'localhost'
local PORT = '8080'

local socket = require("socket")
local tcp = assert(socket.tcp())

tcp:connect(HOST, PORT)
local frame = 0
local input = setInput("000000000")
local next_reset_slot = 5
local max_saved = 1
local all_saved = true
local restart_count = 0

-- savestate.loadslot(5)
while true do
	-- if frame == 0 then
	--send data
	tcp:send(1) --start of data
	r, status, partial = tcp:receive()
	tcp:send(memory.read_u16_le(power))
	r, status, partial = tcp:receive()
	tcp:send(memory.read_s16_le(checkpoint))
	r, status, partial = tcp:receive()
	tcp:send(memory.read_u8(reversed))
	r, status, partial = tcp:receive()
	tcp:send(memory.read_u8(game_over))
	r, status, partial = tcp:receive()
	tcp:send(memory.read_u16_le(speed))

	r, status, partial = tcp:receive()
	client.screenshottoclipboard()
	tcp:send(1)
	-- print("All data sent")

	for i = 1, 4 do
		joypad.set(input, 1)
		emu.frameadvance()
	end
	-- print("Executing action: ", input)

	-- ready for next action
	action, status, partial = tcp:receive()
	-- print(action)
	input = setInput(action)
	if input["A"] == true then
		-- print("Resetting to slot ", next_reset_slot)
		savestate.loadslot(next_reset_slot)
		tcp:send(next_reset_slot)
		-- savestate.loadslot(5)
		-- tcp:send(5)
		-- if (next_reset_slot == 5 or next_reset_slot == 6) and restart_count < 1 then
		-- 	restart_count = restart_count + 1
		-- 	-- print(restart_count)
		-- elseif restart_count >= 1 then
		-- 	restart_count = 0
		-- 	next_reset_slot = (next_reset_slot + 1) % 10
		-- else
		-- 	next_reset_slot = (next_reset_slot + 1) % 10
		-- end
		next_reset_slot = (next_reset_slot+1) % 10
		input = setInput("000000000")
		-- print("Done")
	end

end


tcp:close()
