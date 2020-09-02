import os
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

from PIL import ImageGrab, Image, ImageFilter
from distutils.dir_util import copy_tree
from random import shuffle

class State:
    def __init__(self, image, checkpoint, power, reversed, last_act):
        self.image = image
        self.checkpoint = checkpoint
        self.power = power
        self.reversed = reversed
        self.last_act = last_act

    def view_image(self):
        img = self.image[0]
        img = np.hstack((img[0], img[1], img[2], img[3], img[4], img[5], img[6], img[7]))
        Image.fromarray(img).show()


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        self.value_fc1 = nn.Linear(in_features=256*5 + 5, out_features=1024)
        self.value_out = nn.Linear(in_features=1024, out_features=1)

        self.advantage_fc1 = nn.Linear(in_features=256*5 + 5, out_features=1024)
        self.advantage_fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.advantage_out = nn.Linear(in_features=1024, out_features=5)

    def forward(self, x, last_act):
        # Convolution forward
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 256*5)
        z = torch.cat((x, last_act), dim=1)

        # Value Stream
        v = F.relu(self.value_fc1(z))
        v = self.value_out(v)

        # Advantage Stream
        a = F.relu(self.advantage_fc1(z))
        a = F.relu(self.advantage_fc2(a))
        a = self.advantage_out(a)

        # Final Q-Values
        x = v + a - torch.mean(a, dim=1, keepdim=True)
        return x

class Agent:
    def __init__(self):
        # Hyperparameters
        self.discount = 0.99
        self.epsilon = 0.01
        self.alpha = 0.00001
        self.eps_decay = 0.000005
        self.batch_size = 32

        # Bookkeeping
        self.replay_buffer = []
        self.buffer_size = 100000
        self.last_checkpoint = -198
        self.curr_power = 2048

        # Saving paths
        self.optimizer_path = "model/optimizer.pth"
        self.model_path = "model/model.pth"

        # Additional data
        self.action_input_dict = {0: util.action_to_input(["Left", "B"]),
                                  1: util.action_to_input(["B"]),
                                  2: util.action_to_input(["Right", "B"]),
                                  3: util.action_to_input(["Left",  "L"]),
                                  4: util.action_to_input(["Right", "R"])}
        self.action_dict = {0: "Left",
                            1: "Forward",
                            2: "Right",
                            3: "Drift Left",
                            4: "Drift Right"}

        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)

        # Define cost and optimizer
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.training_network.parameters(), lr=self.alpha)

        # Load model/optimizer parameters if available
        try:
            self.training_network.load_state_dict(torch.load(self.model_path))
            self.target_network.load_state_dict(torch.load(self.model_path))
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            print("Model restored")
        except:
            print("New Training Session")



    def calculate_reward(self, state, game_over, speed):
        power = state.power
        checkpoint = state.checkpoint
        reversed = state.reversed

        if checkpoint >= 1280:
            return 1
        elif reversed == 1 or power < self.curr_power or game_over == 128:
            return -1
        elif speed > 1200:
            return 1
        elif speed > 0:
            return 0
        else:
            return 0

    def observe(self, image, checkpoint, power, reversed, game_over, last_act, speed):
        state = State(torch.FloatTensor(image), checkpoint, power, reversed, torch.FloatTensor(last_act))

        # Calculate reward
        reward = self.calculate_reward(state, game_over, speed)
        self.last_checkpoint = checkpoint
        self.curr_power = power

        # Compute action based on epsilon-greedy policy
        p = np.random.binomial(1, self.epsilon)
        desired_action = int(torch.argmax(self.get_training_action(state)))
        action = desired_action if p == 0 else np.random.randint(0, 5)

        # Add experience to replay_buffer
        is_terminal = True if power < 0 or game_over == 128 or reversed == 1 or checkpoint >= 1280 else False
        self.add_to_buffer(state, action, reward, is_terminal)

        # print("Desired Action: ", self.action_dict[desired_action],
              # "| Actual Action: ", self.action_dict[action],
              # "| Last Action: ", self.action_dict[np.argmax(last_act)])
              # "| Checkpoint: ", checkpoint,
              # "| Power: ", power,
              # "| Reward: ", reward,
              # "| Reversed: ", reversed,
              # "| Game Over: ", game_over,
              # "| Terminal: ", is_terminal,
              # "| Speed: ", speed)

        # print("Reward: ", reward)
        return action, reward

    def add_to_buffer(self,state, action, reward, is_terminal):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer = self.replay_buffer[1:self.buffer_size+1]

        self.replay_buffer.append([state, action, reward, is_terminal])

    def optimize(self, iterations):
        # print("Optimizing")
        for j in range(iterations):
            # get mini-batch
            batch = [np.random.randint(0, len(self.replay_buffer)) for k in range(self.batch_size)]
            images, actions, y = self.batch_to_feed_dict(batch)
            images = images.to(self.device)
            actions = actions.to(self.device)
            y = y.to(self.device)

            # forward pass
            predictions = self.training_network(images, actions)
            loss = self.criterion(predictions, y)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

             # if j % 100 == 0:
                # print("Iteration {}, Loss {}".format(j, loss.item()) )

    def save_model(self):
        torch.save(self.training_network.state_dict(), self.model_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        shutil.rmtree('/backup', ignore_errors=True)
        copy_tree("model", "backup")

    def get_training_action(self, state):
        image = state.image.to(self.device)
        last_act = state.last_act.to(self.device)
        return self.training_network.forward(image, last_act).detach()


    def get_target_action(self, state):
        image = state.image.to(self.device)
        last_act = state.last_act.to(self.device)
        return self.target_network.forward(image, last_act).detach()

    def batch_to_feed_dict(self, batch):
        feed_dict = {}
        x = []
        a = []
        y = []

        for b in batch:
            state0, action0, reward0, terminal0 = self.replay_buffer[b]
            q0 = self.get_training_action(state0).cpu().numpy()
            if terminal0 or b == len(self.replay_buffer) - 1: #Terminal State
                q0[0, action0] = reward0
            else:
                state1, action1, reward1, terminal1 = self.replay_buffer[b+1]
                q1_main = self.get_training_action(state1)[0]
                q1_target = self.get_target_action(state1)[0]
                action = torch.argmax(q1_main)
                q0[0, action0] = reward0 + self.discount * q1_target[action]

            x.append(state0.image.numpy())
            a.append(state0.last_act.numpy())
            y.append(q0)

        x_tup = tuple(i for i in x)
        a_tup = tuple(i for i in a)
        y_tup = tuple(i for i in y)

        return torch.FloatTensor(np.vstack(x_tup)), torch.FloatTensor(np.vstack(a_tup)), torch.FloatTensor(np.vstack(y_tup))

    def update_target(self):
        self.target_network.load_state_dict(self.training_network.state_dict())

    def save_experiences(self):
        with open(self.exp_path, 'wb') as f:
            pickle.dump(self.replay_buffer, f)

    def view_sequence(self, start, end):
        for i in range(start, end):
            self.examine_buffer(i)

    def examine_buffer(self, i):
        state = self.replay_buffer[i][0]
        print("Predicted Q: ", self.get_training_action(state), "Reward: ", self.replay_buffer[i][2], "Terminal: ", self.replay_buffer[i][3])
        state.view_image()
