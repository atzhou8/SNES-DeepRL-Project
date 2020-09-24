import util
import socket
import pickle
import numpy as np
import torch
from PIL import ImageGrab, Image

import deep_q_agent

agent = deep_q_agent.Agent()
HOST = 'localhost'
PORT = 8080

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()
print('Connected by', addr)
frame_count = 0

start_image = util.process_image(Image.open("images/start_image.png"))
start = ([start_image for _ in range(4)], 0) # feed a list of blank images of a dummy checkpoint to start with

max_reward = 0
curr_reward = 0
eps_since_max = 0
ep = 0
images = []
images, agent.curr_checkpoint = start
last_act=np.array([[0,0,0,0,0]])
while True:
    frame_count = (frame_count + 1) % 1000000

    # Receive relevant data
    data = conn.recv(1024)
    conn.sendall("0\n".encode('ascii'))
    power = int(conn.recv(1024))
    conn.sendall("0\n".encode('ascii'))
    checkpoint = int(conn.recv(1024))
    conn.sendall("0\n".encode('ascii'))
    reversed = int(conn.recv(1024))
    conn.sendall("0\n".encode('ascii'))
    game_over = int(conn.recv(1024))
    conn.sendall("0\n".encode('ascii'))
    speed = int(conn.recv(1024))
    conn.sendall("0\n".encode('ascii'))

    # Receive next image
    images = images[1:4]
    r = conn.recv(1024)
    images.append(util.get_image())

    action, reward = agent.observe(np.array([images]), checkpoint, power, reversed, game_over, last_act, speed)
    last_act = np.array([[int(i==action) for i in range(5)]])
    action = agent.action_input_dict[action] + "\n"
    curr_reward += agent.discount * reward

    # if frame_count > 500:
        # break

    if frame_count >= 25000:
        if frame_count % 500 == 0:
            agent.save_model()
        if frame_count % 4 == 0:
            agent.optimize(1)
            agent.epsilon -= agent.eps_decay
            if agent.epsilon <= 0.075:
                agent.epsilon = 0.7
        if frame_count % 2500 == 0:
            agent.update_target()

    # book-keeping at end of episode
    if power < 0 or game_over == 128 or reversed == 1 or checkpoint >= 1280:
        ep = ep + 1
        print("Episode: ", ep, " |Epsilon ", agent.epsilon)
        action = util.action_to_input(["A"]) + "\n"
        conn.sendall(action.encode('ascii'))
        slot_number = int(conn.recv(1024))
        if curr_reward <= max_reward:
            eps_since_max += 1
        else:
            max_reward = curr_reward
            eps_since_max = 0
        print("Reward: ", curr_reward, " |Max Reward: ", max_reward, " |Eps Since Max:", eps_since_max)
        print("---------------------------------------------------------------------------------------")
        # print("Reloading to slot", slot_number)
        images, agent.last_checkpoint = start
        last_act = np.array([[0,0,0,0,0]])
        curr_reward = 0
        continue

    conn.sendall(action.encode('ascii'))
