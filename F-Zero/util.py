from PIL import Image, ImageGrab, ImageDraw, ImageFilter
import numpy as np

def get_image():
    image = None
    while image is None:
        image = ImageGrab.grabclipboard()
    return process_image(image)

def process_image(img, width=84, height=58):
    img = img.convert('L')
    img = clean_image(img)
    img = img.resize((width, height))
    # img = img.filter(ImageFilter.MedianFilter(5))
    # img = img.filter(ImageFilter.MinFilter(5))
    # img = img.filter(ImageFilter.GaussianBlur())
    # img = img.filter(ImageFilter.EDGE_ENHANCE())
    return np.array(img)

def clean_image(img):
    # crop out horizon of track
    img = img.crop((0, 48, 256, 224))

    #remove extraneous
    boost_box = [(208, 160), (232, 168)]
    # timer_box = [(184, 2), (240, 14)]
    # car_box = [(104, 110), (152, 144)]
    drawer = ImageDraw.Draw(img)
    drawer.rectangle(boost_box, fill=166,outline=166)
    # drawer.rectangle(timer_box, fill=166,outline=166)
    # drawer.rectangle(car_box, fill=166,outline=166)

    return img

def action_to_input(input):
    buttons = ["A", "B", "X", "Y", "Up", "Down", "Left", "Right", "Start"]
    action = ""
    for i in range(len(buttons)):
        if buttons[i] in input:
            action += "1"
        else:
            action += "0"

    return action
