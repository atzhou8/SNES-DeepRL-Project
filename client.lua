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
		-- ready for next action
		tcp:send(0)
  	action, status, partial = tcp:receive()
		client.screenshottoclipboard()
		tcp:send(1)
		-- game screen loaded to clipboard
		input = setInput(action)
	end
	joypad.set(input, 1)
	frame = (frame + 1) % 2
	-- print("Frame", frame)
	emu.frameadvance()
end
tcp:close()
