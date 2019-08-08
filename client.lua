Buttons = {
	"A",
	"B",
	"X",
	"Y",
	"Up",
	"Down",
	"Left",
	"Right",
}

local speed = 0x000B20
local power = 0x000E22
local is_reversed = 0x0000B8

function setInput(actionstring)
	input = {}
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
--
-- main loop
local HOST = 'localhost'
local PORT = '8080'

local socket = require("socket")
local tcp = assert(socket.tcp())

tcp:connect(HOST, PORT)
frame = 0
local input = 0
while true do
	if frame == 0 then
		--send data
		client.screenshottoclipboard()
		tcp:send(1) --start of data
		r, status, partial = tcp:receive()
		tcp:send(memory.read_u8(is_reversed))
		r, status, partial = tcp:receive()
		tcp:send(memory.read_u16_le(speed))
		r, status, partial = tcp:receive()
		tcp:send(memory.read_u16_le(power))
		-- ready for next action
  	action, status, partial = tcp:receive()
		input = setInput(action)
	end
	joypad.set(input, 1)
	frame = (frame + 1) % 6
	emu.frameadvance()
end
tcp:close()
