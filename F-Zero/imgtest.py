from PIL import Image, ImageGrab, ImageFilter

resize_height = 16
resize_width = 26


def get_image():
    img = ImageGrab.grabclipboard()
    img = img.convert('L')
    box1 = (0, 9, 5, 16)
    box2 = (20, 13, 25, 16)
    img = img.filter(ImageFilter.MinFilter(7))
    img = img.filter(ImageFilter.GaussianBlur(1))
    img = img.crop((0, 62, 256, 224))
    img = img.resize((resize_width, resize_height))


    c = []
    c.append(img.crop(box1))
    c.append(img.crop(box2))

    for i in range(len(c)):
        ic = c[i]
        for _ in range(2):
            ic = ic.filter(ImageFilter.SMOOTH_MORE)
        ic = ic.filter(ImageFilter.MedianFilter(9))
        # ic = ic.filter(ImageFilter.GaussianBlur(2))
        # for _ in range(1):
        # ic = ic.filter(ImageFilter.GaussianBlur(2))



        c[i] = ic

    img.paste(c[0], box1)
    img.paste(c[1], box2)
    return img

get_image().show()
