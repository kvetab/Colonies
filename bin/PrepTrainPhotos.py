import PIL
from PIL import  Image
import os


w = 1000
h = 1000
directory = r"C:\Users\Kiki\PycharmProjects\ColonyCount\fotky\\"

def ChangeImage(filename):
    im = Image.open(filename)
    img = im.resize((w, h), PIL.Image.ANTIALIAS)
    img.save(filename[0:-4]+".png", "PNG")
    im.close()

def ChangeDir():
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            ChangeImage(directory + filename)

ChangeImage("PICT9572.jpg")