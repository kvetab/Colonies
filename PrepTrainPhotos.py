import PIL
from PIL import  Image
import os


w = 1000
h = 1000
directory = r"C:\Users\Kiki\PycharmProjects\ColonyCount\fotky\\"

def ChangeImage(filename):
    im = Image.open(filename)
    #area = (400, 400, 800, 800)
    #cropped_img = im.crop(area)
    img = im.resize((w, h), PIL.Image.ANTIALIAS)
    img.save(filename[0:-4]+".png", "PNG")
    #print(filename[0:-4]+".png")
    im.close()

def ChangeDir():
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            ChangeImage(directory + filename)

ChangeImage("agar6.png")