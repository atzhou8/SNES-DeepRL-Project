import socket
import deep_q_agent
import tensorflow as tf
import numpy as np
from PIL import ImageGrab

session = tf.Session()
agent = deep_q_agent.Agent(session)
HOST = 'localhost'
PORT = 8080

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()
print('Connected by', addr)
while True:
    data = conn.recv(1024)
    if data == b'1':
        image = agent.get_image()
        conn.sendall("0\n".encode('ascii'))
        is_reversed = float(conn.recv(1024))
        conn.sendall("0\n".encode('ascii'))
        speed = float(conn.recv(1024))
        conn.sendall("0\n".encode('ascii'))
        power = float(conn.recv(1024))

        action = agent.observe(image, speed, power, is_reversed) + "\n"
        conn.sendall(action.encode('ascii'))
        # print("Action sent", action)
