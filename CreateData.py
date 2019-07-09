# take an image + coords file, load them
# create set amount of random squares from the image
# count colonies in each square
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
    coords = LoadCoords(coordsf)

    img = Image.open(imgfile)
    arr = np.array(img)
    h, w = arr.shape[:2]

    samples = []
    labels = []
    for i in range(n):
        x = random.randint(0,h-65)
        y = random.randint(0,w-65)
        size = random.randint(60,240)
        while x+size > h or y+size > w:
            size = random.randint(60,240)
        square, label = GetSquare(img, coords, x, y, size)
        sample = square.reshape(28812)
        samples.append(sample)
        labels.append(label)

    SaveToFile(imgfile, samples, labels)

# loads list of coordinates
def LoadCoords(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)
    for line in coords:
        if len(line) <1:
            coords.remove(line)
    coords.sort(key=sortSecond)     #maybe unnecessary
    coords.sort(key=sortFirst)
    coords2 = []
    for line in coords:
        i = float(line[0])
        j = float(line[1])
        k = int(line[2])
        coords2.append((i,j,k))
    return coords2

def CreateSpecifSample(x, y, size, imfile, coordfile):
    coords = LoadCoords(coordfile)
    img = Image.open(imfile)
    arr = np.array(img)
    square, label = GetSquare(img, coords, x, y, size)
    area = (x, y, x+size, y+size)
    res = img.crop(area)
    res.show()
    sample = square.reshape(28812)
    SaveToFile(imfile, [sample], [label])

def SaveToFile(imgfile, samples, labels):
    outf = "out" + imgfile[0:-4] + ".txt"
    with open(outf, 'a', newline='') as f:
        writer = csv.writer(f)
        for line in samples:
            writer.writerow(line)
    outf = "lab" + imgfile[0:-4] + ".txt"
    with open(outf, 'a', newline='') as f:
        writer = csv.writer(f)
        for line in labels:
            writer.writerow(str(line))


def GetSquare(img, coords, x, y, size):
    area = (x, y, x+size, y+size)
    cropped_img = img.crop(area)
    im = cropped_img.resize((98, 98), Image.ANTIALIAS)
    sq = np.array(im)
    i = 0
    count = 0
    while coords[i][0] < x:
        i+=1
    while coords[i][0] >= x and coords[i][0] <= x+size:
        if coords[i][1] >= y and coords[i][1] <= y + size and coords[i][2] == 1:
            count += 1
        i += 1
    return sq, count
    # I'm confused how the x and y coords should be... -> probably right now

CreateNSamples(30, "PICT9566.png", "coords9566.csv")
#CreateSpecifSample(600, 135, 110, "PICT9563.png", "coords9563.csv")

"""
img = Image.open("PICT9563.png")
#area = (350, 150, 500, 300)
#res = img.crop(area)
#res.show()

with open("coords9563.csv", 'r') as f:
    reader = csv.reader(f)
    coords = list(reader)
    # loads list of coordinates

for line in coords:
    if len(line) < 1:
        coords.remove(line)
coords.sort(key=sortSecond)  # maybe unnecessary
coords.sort(key=sortFirst)
coords2 = []
for line in coords:
    i = float(line[0])
    j = float(line[1])
    k = int(line[2])
    coords2.append((i, j, k))
sq, c = GetSquare(img, coords2, 350, 150, 150)
print(c)

"""



