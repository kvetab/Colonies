# take an image + coords file, load them
# create set amount of random squares from the image
# count colonies in each square
# save image (square) in one file, label in another
COORDS_DCT = "coords/"
IMG_DCT = "photos_used/"

import numpy as np
import csv
from PIL import Image, ImageDraw
import random

import os

def sortSecond(val):
    return val[1]
def sortFirst(val):
    return val[0]

def ProcessDirectory(IMG_DCT, COORDS_DCT):
    images = [f for f in os.listdir(IMG_DCT) if os.path.isfile(os.path.join(IMG_DCT, f))]
    try:
        #os.remove("labels/labels.csv")
        i = 0
    except:
        pass
    for imgf in images:
        #verify whether corresponding coords file exists
        coordf = imgf.replace("PICT","coords").replace("png","csv")
        if os.path.isfile(os.path.join(COORDS_DCT, coordf)):
            CreateNSamples(250, imgf, coordf, IMG_DCT, COORDS_DCT)

def CreateNSamples(n, imgfile, coordsf, IMG_DCT, COORDS_DCT):
    coords = LoadCoords(coordsf, COORDS_DCT)

    img = Image.open(IMG_DCT + imgfile)
    arr = np.array(img)
    h, w = arr.shape[:2]

    samples = []
    labels = []
    images = []
    for i in range(n):
        w_crop = random.randint(75, 110)
        h_crop = random.randint(75, 110)

        x = random.randint(0,h-h_crop)
        y = random.randint(0,w-w_crop)
        square, label, img_crop = GetSquare(img, coords, x, y, h_crop, w_crop)
        labels.append(label)
        images.append(img_crop)

    SaveToFile(imgfile, samples, labels, images)

# loads list of coordinates
def LoadCoords(filename, COORDS_DCT):


    with open(COORDS_DCT+filename, 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)
    for line in coords:
        if len(line) <1:
            coords.remove(line)
    coords2 = []
    for line in coords:
        try:
            i = float(line[0])
            j = float(line[1])
            k = int(line[2])
            coords2.append((i,j,k))
        except:
            #LP: add some error here (incorrect line appears)
            pass

    return coords2


# not tested
def CreateSpecifSample(x, y, size, imfile, coordfile):
    coords = LoadCoords(coordfile, COORDS_DCT)
    img = Image.open(imfile)
    arr = np.array(img)
    square, label, crop = GetSquare(img, coords, x, y, size, size)
    area = (x, y, x+size, y+size)
    res = img.crop(area)
    res.show()
    sample = square.reshape(28812)
    SaveToFile(imfile, [sample], [label], [crop])



def SaveToFile(imgfile, samples, labels, images):
    imgf = "crop_"+ imgfile[0:-4] +"_"

    labelsf = "labels.csv"
    with open("new_photos/labels/"+labelsf, 'a', newline='') as f:
        writer = csv.writer(f)
        for count,img in enumerate(images):
            name = imgf + str(count) + ".png"
            img.save("new_photos/test_crops/"+name)
            writer.writerow((name,str(labels[count])))



def GetSquare(img, coords, x, y, h_crop, w_crop):
    area = (x, y, x+h_crop, y+w_crop)
    cropped_img = img.crop(area)
    cropped_img.convert('RGB')
    im = cropped_img.resize((98, 98), Image.ANTIALIAS)
    sq = np.array(im)
    print(x, y, x+h_crop, y+w_crop)
    res_coords = [c for c in coords if (c[0] >=x and c[0] < x + h_crop and c[1] >= y and c[1] < y + w_crop and c[2] == 1)]
    print(res_coords)
    count = len(res_coords)
    return sq, count, im




if __name__ == "__main__":
    ProcessDirectory("new_photos/", COORDS_DCT)




