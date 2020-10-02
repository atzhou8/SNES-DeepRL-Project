# SNES-DeepRL-Project

For this project, I used Pytorch to develop a reinforcement learning agent to play the SNES game F-Zero using a double deep Q network with dueling architecture. The game was run on BizHawk, an emulator with Lua scripting capabilities. I then interfaced this with Python using socket programming. 

![Alt Text](https://github.com/azhou314/SNES-DeepRL-Project/blob/master/F-Zero/gifs/example1.gif)
![Alt Text](https://github.com/azhou314/SNES-DeepRL-Project/blob/master/F-Zero/gifs/example2.gif)
![Alt Text](https://github.com/azhou314/SNES-DeepRL-Project/blob/master/F-Zero/gifs/example3.gif)

## Problem Formulation
### States, Actions, and Reward
The states of the model are a combination of downsampled image frames from the game's grayscale output, and the last action that the agent chose. One game frame is sampled out of every four, and the agent is provided with the four most recent samples. I played around with how often to sample and how many samples to give to the agent, but landed on this as a good tradeoff between speed and consistency.

The agent has a discrete action space consisting of five actions: accelerate, accerelate left, accelerate right, drift left, and drift right. The agent originally only had access to the first 3 actions, but I later discovered that most of the more complex tracks cannot be completed unless the agent is able to drift without accelerating.

I also played around with many reward functions, including tinkering with RAM values to extract the agent's position on the map to infer its progress through the track. In the end however, I found that a very simple reward function performed the best: a reward of 1 for going at or above a target speed and for completing the track, and a reward of -1 for running into a wall or dying.

### Network architecture

The network architecture begins with a series of convolutional layers that are applied to the input frames. The output of the convolutional layers is then flattened and concatenated with the agent's last action as a one-hot encoded vector. This flattened and concatenated output is then passed through two streams: a value stream that aims to estimate the relative "goodness" of the agent's current state and a separate advantage stream that evaluate the value of each action. The value and advantage values are then combined to create the Q-value estimate.

Huber loss was used to clip the error term, and optimization was done using the Adam algorithm.

### Training Regime

The agent was allowed to drive on 3 tracks, with 2 or 3 predefined locations for each track. It was able to explore until its health goes below a predefined threshold (about 75%), upon which the agent is given a terminal state and sent to the next episode. This was to prevent the agent from getting stuck on sides of the track, which often happened if it was allowed to explore further. Observations were stored in a replay buffer, which was randomly sampled from during optimization.

Exploration was done with an epsilon-greedy policy. The agent started with an epsilon of 1, with which it explored the environment until it had 25000 observations.
Afterwards, the agent would perform one mini-batch optimization of 32 samples from the replay buffer every 4 observations, and update the target network every 500 observations. The epsilon would be decayed with every optimization, bouncing back to around 0.7 if it grew too low.

Following the double Q learning technique, I maintained a training network to be optimized and a target network to update with the training network's weights periodically during training. When estimating Q-values for optimization, rather than combining the observed reward at one state with the training network's maximum Q-estimate in the next state, the Bellman equation instead combines the reward with the target network's Q-value estimate at the training network's guess for the best action.

Training took place over around 2,000,000 episodes on a laptop GTX 1060, taking about 3 days.

## Performance
Agent driving on the track it was trained on three of the tracks that it was trained on:

![Alt Text](https://github.com/azhou314/SNES-DeepRL-Project/blob/master/F-Zero/gifs/example1.gif)
![Alt Text](https://github.com/azhou314/SNES-DeepRL-Project/blob/master/F-Zero/gifs/example2.gif)
![Alt Text](https://github.com/azhou314/SNES-DeepRL-Project/blob/master/F-Zero/gifs/example3.gif)

Agent attempting to generalize on one of the final tracks, which introduces the additional complexity of having "winds" that force the agent to move in a certain direction:

![Alt Text](https://github.com/azhou314/SNES-DeepRL-Project/blob/master/F-Zero/gifs/example4.gif)

I'm pretty happy with the performance of the agent, but there are a few limitations to discuss. Because of the agent only has access to the last 4 frames, it sometimes is unable to tell which way  is forward on the track. While this is usually not a problem, there are a couple of very deep turns on some of the maps that the agent has trouble with. In addition, because the reward function only awards going at a target speed, it doesn't directly affect the agent's total time. Once the agent is able to successfully drive through a track, it no longer has any incentive to try and complete it faster.

Overall, I really enjoyed this project, and am definitely planning on making improvements in the future!
