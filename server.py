import socket
import numpy as np
from PIL import ImageGrab

def getAction(input_list):
    buttons = ["A", "B", "X", "Y", "Up", "Down", "Left", "Right"]
    action = ""
    for i in range(len(buttons)):
        if buttons[i] in input_list:
            action += "1"
        else:
            action += "0"

    return action

HOST = 'localhost'
PORT = 8080

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()
print('Connected by', addr)
while True:
    data = conn.recv(1024)
    if data == b'0':
        action = getAction(["B", "A", "Left"]) + "\n"
        conn.sendall(action.encode('ascii'))
    elif data == b'1':
        image = np.array(ImageGrab.grabclipboard())
        print(image)
