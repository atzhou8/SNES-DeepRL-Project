import numpy as np
import tensorflow as tf
from PIL import ImageGrab, Image

image_height = 100
image_width = 88

def input_to_action(input_list):
    buttons = ["A", "B", "X", "Y", "Up", "Down", "Left", "Right"]
    action = ""
    for i in range(len(buttons)):
        if buttons[i] in input_list:
            action += "1"
        else:
            action += "0"

    return action

class Network:
    def __init__(self):
        self.alpha = 1e-4

        self.image_flattened = image_height * image_width
        self.conv1_filter_size = 5
        self.conv1_out_channels = 32
        self.conv2_filter_size = 5
        self.conv2_out_channels = 64

        self.x = tf.placeholder(tf.float32, shape=[None, self.image_flattened], name='x')
        self.x_img = tf.reshape(self.x, [-1, image_height, image_width, 1])

        self.y = tf.placeholder(tf.float32, shape=[1,1])
        self.conv1, self.w1, self.b1 = self.create_conv(self.x_img, self.conv1_filter_size, 1, self.conv1_out_channels)
        self.conv2, self.w2, self.b2 = self.create_conv(self.conv1, self.conv2_filter_size, self.conv1_out_channels, self.conv2_out_channels)

        conv_flattened_size = image_height * image_width * self.conv2_out_channels
        self.conv2_flattened = tf.reshape(self.conv2, [-1, conv_flattened_size ])

        self.fc3, self.w3, self.b3 = self.create_fc(self.conv2_flattened, conv_flattened_size, 3)

        self.loss = tf.square(self.y - tf.reduce_max(self.fc3))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss)

    def create_conv(self, input, filter_size, in_channels, out_channels):
        filter_shape = [filter_size, filter_size, in_channels, out_channels]
        weights = tf.Variable(tf.random.truncated_normal(filter_shape))
        bias = tf.Variable(tf.random.truncated_normal([out_channels]))

        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        layer = tf.nn.relu(layer + bias)
        return layer, weights, bias

    def create_fc(self, input, num_inputs, num_outputs):
        weights = tf.Variable(tf.random.truncated_normal([num_inputs, num_outputs]))
        bias = tf.Variable(tf.random.truncated_normal([num_outputs]))
        layer = tf.nn.relu(tf.matmul(input, weights) + bias)

        return layer, weights, bias

class Agent:
    def __init__(self, session):
        self.network = Network()
        self.discount = 0.5
        self.batch_size = 50
        self.replay_buffer = []
        self.session = session
        self.action_input_dict = {0: input_to_action(["Left", "B"]),
                                  1: input_to_action(["B"]),
                                  2: input_to_action(["Right", "B"])}

        #default data for calculating reward
        self.curr_speed = 0
        self.curr_power = 60592

        self.session.run(tf.global_variables_initializer())


    def calculate_reward(self, speed, power, is_reversed):
        a = -1 if is_reversed == 128 else 1
        b = speed - self.curr_speed
        c = power / self.curr_power
        self.curr_speed = speed
        self.curr_power = power
        return a * (b + c)

    def observe(self, image, speed, power, is_reversed):
        s = image
        r = self.calculate_reward(speed, power, is_reversed)
        a, q = self.get_action(s)

        self.replay_buffer.append((s, r, a))
        return a

    def optimize(self, iterations):
        for _ in range(iterations):
            batch = [np.random.randint(0, len(self.replay_buffer)) for k in range(self.batch_size)]
            self.session.run(self.network.optimizer, feed_dict=self.batch_to_feed_dict(batch))

    def get_action(self, image):
        actions = self.session.run(self.network.fc3, feed_dict={self.network.x: image})
        action = np.argmax(actions)
        value = np.max(actions)
        return self.action_input_dict[action], value

    def batch_to_feed_dict(self, batch):
        feed_dict = {}
        for b in batch:
            s1, r1, a1 = self.replay_buffer[b]
            if b == len(self.replay_buffer) - 1:
                feed_dict[self.network.x] = s1
                feed_dict[self.network.y] = r1
            else:
                s2, r2, a2 = self.replay_buffer[b+1]
                _, q = self.get_action(s2)
                feed_dict[self.network.x] = s1
                feed_dict[self.network.y] = r1 + self.discount * q

        return feed_dict

    def process_image(self, image):
        image = image.resize((image_width, image_height)).convert('L')
        return np.array(image).reshape((1,8800))

    def get_image(self):
        # TODO: grabclipboard() sometimes fails. Try-catch temporary solution 
        image = None
        while image is None:
            try:
                image = ImageGrab.grabclipboard()
            except:
                pass


        return self.process_image(image)
