import pickle
import numpy as np
from tkinter.filedialog import askopenfilename
import csv
from PIL import Image
import Utils
import LoadImage


# v tuto chvili to vezme soubor souradnic a predela ho to na soubor "puvodnich" divnoudaju do kNN

def GetData(filename, win, num, imagef):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)
#    imagef = str.strip(coords[0][0])

    img = Image.open(imagef)
    arr = np.array(img)
    h, w = arr.shape[:2]
    mask = LoadImage.createCircularMask(h, w)
    smallMask = LoadImage.createCircularMask(win, win)
    r = int(win/2)

    meanR = arr[:, :, 0][mask].mean()
    meanG = arr[:, :, 1][mask].mean()
    meanB = arr[:, :, 2][mask].mean()

    dataset = {}
    dataset['meanR'] = meanR
    dataset['meanB'] = meanB
    dataset['meanG'] = meanG
    values = []

    def getSquare(x,y,value):
        square = arr[int(y-r): int(y + r + 1), int(x - r): int(x + r + 1), :]
        if square.shape == (17,17,3):
            return Utils.getValues(square, meanR, meanG, meanB, smallMask, x, y, num, value)
        else:
            return [0]

    def getSquareFull(x, y, value):
        square = arr[int(y-r): int(y + r + 1), int(x - r): int(x + r + 1), :]
        return  (x, y, square, value)

    #del coords[0]
    #num = int(coords[0][0])
    #del coords[0]
    for point in coords:
        if len(point)>0:
            lin = getSquare(float(point[0]), float(point[1]), float(point[2]))
            if len(lin)>1:
                values.append(lin)
            #values.append(getSquare(float(point[0]), float(point[1]), float(point[2])))

#    dataset['values'] = values
    #dataset['imgnum'] = num

    outf = 'colemp' + str(num) + '.csv'
#    with open(outf, 'wb') as f:
#        pickle.dump(dataset, f)
    with open(outf, 'a') as f:
        writer = csv.writer(f)
        for line in values:
            writer.writerow(line)

GetData('coords9570.csv', 17, 9570, "PICT9570.png")

