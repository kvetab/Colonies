# take an image + coords file, load them
# create set amount of random squares from the image
# count clonies in each square
# save array (square) in one file, label in another

import numpy as np
import csv
from PIL import Image
import random


def sortSecond(val):
    return val[1]
def sortFirst(val):
    return val[0]

def CreateNSamples(n, imgfile, coordsf):
    with open(coordsf, 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)
        # loads list of coordinates

    coords.sort(key=sortSecond)     #maybe unnecessary
    coords.sort(key=sortFirst)

    img = Image.open(imgfile)
    arr = np.array(img)
    h, w = arr.shape[:2]

    for i in range(n):
        x = random.randint(0,h-65)
        y = random.randint(0,w-65)
        size = random.randint(60,240)
        while x+size > h or y+size > w:
            size = random.randint(60,240)
        square, label = GetSquare(arr, coords, x, y, size)


def GetSquare(img, coords, x, y, size):
    sq = img[x : x + size, y : y + size, :]
    i = 0
    count = 0
    while coords[i][0] < x:
        i+=1
    while coords[i][0] >= x and coords[i][0] <= x+size:
        if coords[i][1] >= y and coords[i][1] <= y + size:
            count += 1
        i += 1
