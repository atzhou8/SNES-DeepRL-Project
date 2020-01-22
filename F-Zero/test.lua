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
}

local checkpoint = 0x001162
local power = 0x000E22
local start = 0x00000C
local reversed = 0x0000B8
local xPos = 0x000B70
local yPos = 0x000B90


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

input = setInput("010000000")
while true do
  joypad.set(input, 1)
  emu.frameadvance()
	print("v")
	print("Start", memory.read_u8(start))
	print("Checkpoint", memory.read_u16_le(checkpoint))
	print("Power", memory.read_u16_le(power))
	print("Reversed", memory.read_u16_le(reversed))
	print("xPos", memory.read_u16_le(xPos))
	print("yPos", memory.read_u16_le(yPos))

end
