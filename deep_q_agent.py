import numpy as np
import tensorflow as tf
from PIL import ImageGrab, Image




class Network:
    def __init__(self):
        self.image_height = 100
        self.image_width = 88
        self.image_flattened = self.image_height * self.image_width

        self.conv1_filter_size = 5
        self.conv1_out_channels = 32
        self.conv2_filter_size = 5
        self.conv2_out_channels = 64

        self.x = tf.placeholder(tf.float32, shape=[None, self.image_flattened])
        self.x_img = tf.reshape(self.x, [-1, self.image_height, self.image_width, 1])

        self.conv1, self.w1, self.b1 = self.create_conv(self.x_img, self.conv1_filter_size, 1, self.conv1_out_channels)
        self.conv2, self.w2, self.b2 = self.create_conv(self.conv1, self.conv2_filter_size, self.conv1_out_channels, self.conv2_out_channels)
        conv_flattened_size = self.image_height * self.image_width * self.conv2_out_channels
        self.conv2_flattened = tf.reshape(self.conv2, [-1, conv_flattened_size ])

        self.fc3, self.w3, self.b3 = self.create_fc(self.conv2_flattened, conv_flattened_size, 3)
        # self.fc1 = self.create_fc(self.conv2_flattened, )

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

    def process_image(self, image):
        image = image.resize((self.image_width, self.image_height)).convert('L')
        return np.array(image)

    def get_image(self):
        return self.process_image(ImageGrab.grabclipboard())

n = Network()
