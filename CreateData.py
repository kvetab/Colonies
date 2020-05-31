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

    #LP: testovani coords file, pro finalni verzi je treba zakomentovat
    #draw = ImageDraw.Draw(img)
    #for c in coords:
    #    draw.ellipse([(c[0]-1, c[1]-1),(c[0]+1, c[1]+1)], "blue", "blue")
    #img.save("test.png")

    samples = []
    labels = []
    images = []
    for i in range(n):
        #LP: trochu upravuju zdrojak pro generovani samples aby byl vic variabilni a vracel obrazky
        # - nejprve to vybere nahodne vysku a sirku vyrezu
        # - pak nahodne vybere souradnici top-left bodu (tak aby to vyslo), vezme vyrez a resizne do 98x98 (to uz jsi tam mela)
        # - nakonec to ulozi vyrez jako obrazek

        w_crop = random.randint(75, 110)
        h_crop = random.randint(75, 110)

        x = random.randint(0,h-h_crop)
        y = random.randint(0,w-w_crop)
        #size = random.randint(60,240)
        size = 98
        # what??
        #while x+size > h or y+size > w:
            #size = random.randint(60,240)
        square, label, img_crop = GetSquare(img, coords, x, y, h_crop, w_crop)
        # square se pak vlastne nikde nepouzije
        #sample = square.reshape(28812)
        #samples.append(sample)
        labels.append(label)
        images.append(img_crop)

    SaveToFile(imgfile, samples, labels, images)
    # samples neni potreba

# loads list of coordinates
def LoadCoords(filename, COORDS_DCT):


    with open(COORDS_DCT+filename, 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)
    for line in coords:
        if len(line) <1:
            coords.remove(line)
#    coords.sort(key=sortSecond)     #maybe unnecessary
#    coords.sort(key=sortFirst)
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
    #outf = "out" + imgfile[0:-4] + ".txt"
    imgf = "crop_"+ imgfile[0:-4] +"_"
    #with open(outf, 'a', newline='') as f:
    #    writer = csv.writer(f)
    #    for line in samples:
    #        writer.writerow(line)
    #outf = "lab" + imgfile[0:-4] + ".txt"
    #with open(outf, 'a', newline='') as f:
    #    writer = csv.writer(f)
    #    for line in labels:
    #        writer.writerow(str(line))


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
    i = 0
    count = 0
    print(x, y, x+h_crop, y+w_crop)
    res_coords = [c for c in coords if (c[0] >=x and c[0] < x + h_crop and c[1] >= y and c[1] < y + w_crop and c[2] == 1)]
    # tady chybelo testovani pozitivni kolonie (jako zakomentovany radek 160)
    print(res_coords)
    count = len(res_coords)
    #LP: chapu to dobre tak, ze jsi nejdriv setridila coords a pak se snazis o trochu inteligentnejsi 2D filtrovani? Nejsem si jisty, co tady dela to i
    # - vzhledem k poctu coords mi to prijde trochu zbytecne, ale asi je to spravne, takze proc ne...
    # - pridam jeden vystupni objekt: samotny obrazek k ulozeni
    """
    soucasne_x = coords[i][0]
    while soucasne_x < x:
        i+=1
        soucasne_x = coords[i][0]
    while i < len(coords) and coords[i][0] >= x and coords[i][0] <= x+size:
        if coords[i][1] >= y and coords[i][1] <= y + size and coords[i][2] == 1:
            count += 1
        i += 1
    """
    return sq, count, im
    # I'm confused how the x and y coords should be... -> probably right now




if __name__ == "__main__":
    ProcessDirectory("new_photos/", COORDS_DCT)
    #CreateSpecifSample(600, 135, 110, "PICT9563.png", "coords9563.csv")
    #CreateNSamples(20, "PICT9563.png", "coords9563.csv", "photos_used/", "coords/")




