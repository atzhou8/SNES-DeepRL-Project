import os
import pickle
import shutil
import numpy as np
import tensorflow as tf
import util

from PIL import ImageGrab, Image, ImageFilter
from distutils.dir_util import copy_tree
from random import shuffle

resize_height = 84
resize_width = 84



class Network:
    def __init__(self, scope):
        self.alpha = 0.00001

        with tf.variable_scope(scope):
            # Inputs
            self.image_input = tf.placeholder(shape=[None, 84, 58, 4], dtype=tf.float32)
            self.last_act = tf.placeholder(shape=[None, 3], dtype=tf.float32)

            # Expected Q-values for Loss
            self.y = tf.placeholder(tf.float32, shape=[None, 3])

            # Convolutional Layers
            self.scaled_image = self.image_input / 255
            self.conv1 = tf.layers.conv2d(
                inputs=self.scaled_image, filters=64, kernel_size=[8, 8], strides=4,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2, filters=128, kernel_size=[3, 3], strides=1,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')
            self.conv4 = tf.layers.conv2d(
                inputs=self.conv3, filters=256, kernel_size=[3, 3], strides=1,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="same", activation=tf.nn.relu, use_bias=False, name='conv4')
            self.conv5 = tf.layers.conv2d(
                inputs=self.conv4, filters=256, kernel_size=[3, 3], strides=1,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False, name='conv5')

            # Dueling Architecture
            self.flattened = tf.concat([tf.contrib.layers.flatten(self.conv5), self.last_act], 1)
            self.value_out = tf.layers.dense(inputs=self.flattened, units=1,
                                             kernel_initializer=tf.variance_scaling_initializer(scale=2))
            self.advantage_fc1 = tf.layers.dense(inputs=self.flattened, units=1024, activation=tf.nn.relu,
                                                 kernel_initializer=tf.variance_scaling_initializer(scale=2))
            self.advantage_fc2 = tf.layers.dense(inputs=self.advantage_fc1, units=1024, activation=tf.nn.relu,
                                                 kernel_initializer=tf.variance_scaling_initializer(scale=2))
            self.advantage_out = tf.layers.dense(inputs=self.advantage_fc2, units=3,
                                                 kernel_initializer=tf.variance_scaling_initializer(scale=2))


            # Network output
            self.out= self.value_out + (self.advantage_out - tf.reduce_mean(self.advantage_out, reduction_indices=1, keepdims=True))

            # Optimization and loss
            self.loss = tf.reduce_mean(tf.losses.huber_loss(labels= self.y, predictions=self.out))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss)


class State:
    def __init__(self, image, checkpoint, power, reversed, last_act):
        self.image = image
        self.checkpoint = checkpoint
        self.power = power
        self.reversed = reversed
        self.last_act = last_act

    def view_image(self):
        img = np.transpose(self.image[0])
        img = np.hstack((img[0], img[1], img[2], img[3]))
        Image.fromarray(img).show()


class Agent:
    def __init__(self, session):
        # Hyperparameters
        self.discount = 0.99
        self.epsilon = 0.01
        self.eps_decay = 0.000005
        self.batch_size = 32

        # Bookkeeping
        self.replay_buffer = []
        self.buffer_size = 1000000
        self.last_checkpoint = -198
        self.curr_power = 2048

        # Saving paths
        self.exp_path = "data/replaydata.pkl"
        self.model_name = "model/model.ckpt"

        # Additional data
        self.action_input_dict = {0: util.action_to_input(["Left", "B"]),
                                  1: util.action_to_input(["B"]),
                                  2: util.action_to_input(["Right", "B"])}
        self.action_dict = {0: "Left",
                            1: "Forward",
                            2: "Right"}

        # Initialize networks and session
        self.training_network = Network('training')
        self.target_network = Network('target')
        self.saver = tf.train.Saver()
        self.session = session
        self.terminal_image = np.array([[0 for i in range(resize_width * resize_height * 4)]])

        try:
            self.saver.restore(self.session, self.model_name)
            print("Model restored")
        except:
            self.session.run(tf.global_variables_initializer())
            # self.update_target()
            print("Initialized new model")

    def calculate_reward(self, state, game_over, speed):
        power = state.power
        checkpoint = state.checkpoint
        reversed = state.reversed

        if reversed == 1 or power < self.curr_power or checkpoint < self.last_checkpoint or game_over == 128:
            return -1
        elif speed > 1200:
            return 1
        elif speed > 0:
            return 0.5
        else:
            return 0

    def observe(self, image, checkpoint, power, reversed, game_over, last_act, speed):
        state = State(image, checkpoint, power, reversed, last_act)

        # Calculate reward
        reward = self.calculate_reward(state, game_over, speed)
        self.last_checkpoint = checkpoint
        self.curr_power = power

        # Compute action based on epsilon-greedy policy
        p = np.random.binomial(1, self.epsilon)
        desired_action = np.argmax(self.get_training_action(state))
        action = desired_action if p == 0 else np.random.randint(0, 3)

        # Add experience to replay_buffer
        is_terminal = True if power < 1500 or game_over == 128 or reversed == 1 else False
        self.add_to_buffer(state, action, reward, is_terminal)

        print("Desired Action: ", self.action_dict[desired_action],
              "| Actual Action: ", self.action_dict[action],
              "| Last Action: ", self.action_dict[np.argmax(last_act)],
              "| Checkpoint: ", checkpoint,
              "| Power: ", power,
              "| Reward: ", reward,
              "| Reversed: ", reversed,
              "| Game Over: ", game_over,
              "| Terminal: ", is_terminal,
              "| Speed: ", speed)

        return action, reward

    def add_to_buffer(self,state, action, reward, is_terminal):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer = self.replay_buffer[1:self.buffer_size+1]

        self.replay_buffer.append([state, action, reward, is_terminal])

    def optimize(self, iterations):
        print("Optimizing")
        for j in range(iterations):
            # batch = [i for i in range(20)]
            batch = [np.random.randint(0, len(self.replay_buffer)) for k in range(self.batch_size)]
            _, loss = self.session.run([self.training_network.optimizer, self.training_network.loss], feed_dict=self.batch_to_feed_dict(batch))
            if j % 100 == 0:
                print("Avg Loss: ", np.mean(loss), "Iteration: ", j)
                # self.update_target()

    def save_model(self):
        saved_path = self.saver.save(self.session, self.model_name)
        shutil.rmtree('/backup', ignore_errors=True)
        copy_tree("model", "backup")

    def get_training_action(self, state):
        image = state.image
        return self.session.run(self.training_network.out, feed_dict={self.training_network.image_input: image,
                                                                      self.training_network.last_act: state.last_act})

    def get_target_action(self, state):
        image = state.image
        return self.session.run(self.target_network.out, feed_dict={self.target_network.image_input: image,
                                                                    self.target_network.last_act: state.last_act})

    def batch_to_feed_dict(self, batch):
        feed_dict = {}
        x = []
        a = []
        y = []

        for b in batch:
            state0, action0, reward0, terminal0 = self.replay_buffer[b]
            q0 = self.get_training_action(state0)
            if terminal0 or b == len(self.replay_buffer) - 1: #Terminal State
                # q0[0, 0] = reward0
                # q0[0, 1] = reward0
                # q0[0, 2] = reward0
                q0[0, action0] = reward0
            else:
                state1, action1, reward1, terminal1 = self.replay_buffer[b+1]
                q1_main = self.get_training_action(state1)[0]
                q1_target = self.get_target_action(state1)[0]
                action = np.argmax(q1_main)
                q0[0, action0] = reward0 + self.discount * q1_target[action]

            x.append(state0.image)
            a.append(state0.last_act)
            y.append(q0)

        x_tup = tuple(i for i in x)
        a_tup = tuple(i for i in a)
        y_tup = tuple(i for i in y)

        feed_dict[self.training_network.image_input] = np.vstack(x_tup)
        feed_dict[self.training_network.y] = np.vstack(y_tup)
        feed_dict[self.training_network.last_act] = np.vstack(a_tup)
        return feed_dict

    def update_target(self):
        training_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training')
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        assign_ops = []
        for training, target in zip(training_vars, target_vars):
            assign_ops.append(target.assign(training.value()))
            # print("Assigned", training, "->", target )

        self.session.run(assign_ops)
        print("Updated Target Network")

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
