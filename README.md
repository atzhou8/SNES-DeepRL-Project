# SNES-DeepRL-Project

For this project, I used socket programming and the Bizhawk emulator to develop a reinforcement learning agent to play the SNES game F-Zero with dueling double deep Q network. The game was formulated as a MDP with a continuous state space, involving the last 4 frames of screen input and the agent's last action, and a discrete action space consisting of 3 actions: accelerate, accelerate while turning left, and accelerate while turning right. The agent received a negative award every time it hit a wall, a positive award for going at top speed, and a small positive award for moving at sub-par speeds. 

The network architecture consists of an image input, which after a series of convolutional layers is flattened and concateneated with a one-hot encoded vector of the agent's last action. This is then passed through a dueling architecture, which separately estimates the value and advantage values, before combining them to estimate the Q-values at the given state. Both a target and training network were built using this architecture, and the double Q learning scheme was used for parameter updates.

Training was done locally on a personal laptop with a mobile NVIDIA GTX 1060, over the course of a few days. The agent can clear laps on the map it was trained on, while somewhat clumsily generalizing to the other tracks in the game. 

Agent Driving on the Track it was trained on:
![Alt Text](https://github.com/azhou314/SNES-DeepRL-Project/blob/master/F-Zero/example.gif)
