import numpy as np
from sklearn.neighbors import NearestNeighbors
import LoadImage
import sklearn
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image
import  math
import pandas
import Utils
from PIL import ImageDraw
import csv
import ProbabilityMap as pb

def ChooseInputData(root):
    trdata = np.ndarray(shape=(12, ))
    tsdata = np.ndarray(shape=(12, ))
    def OpenData(data):
        file = askopenfilename(parent=root, initialdir="M:/", title='Choose a file.')
        data1 = np.genfromtxt(file, delimiter=",", skip_header=1)
        data = np.vstack((data1, data))
        return data
    def SplitData(trdata, tsdata):
        file = askopenfilename(parent=root, initialdir="M:/", title='Choose a file.')
        data1 = np.genfromtxt(file, delimiter=",", skip_header=1)
        len = data1.shape[0] / 3
        trdata = np.vstack((data1[:len,:],trdata))
        tsdata = np.vstack((data1[len:2*len,:], tsdata))
        trdata = np.vstack((data1[2*len:,:], trdata))
        return trdata, tsdata

    num = 1
    while (num != '4'):
        num = input('1 for entering train data, 2 for test data, 3 for splitting into train and test, 4 for end.')
        if (num == '1'):
            trdata = OpenData(trdata)
        elif (num == '2'):
            tsdata = OpenData(tsdata)
        elif (num == '3'):
            trdata, tsdata = SplitData(trdata, tsdata)
    return trdata, tsdata

def OpenData():     #opens files listed in settings.txt and loads values for NN, step and window size
    trdata = np.ndarray(shape=(12,))
    tsdata = np.ndarray(shape=(12,))
    file = open("settings.txt", 'r')
    str = file.readline()
    str = file.readline()
    paths = str.split(',')
    str = file.readline()
    for name in paths:
        data = np.genfromtxt(name.strip(), delimiter=",", skip_header=1)
        trdata = np.vstack((data, trdata))
    paths = str.split(',')
    for name in paths:
        data = np.genfromtxt(name.strip(), delimiter=",", skip_header=1)
        tsdata = np.vstack((data, tsdata))
    nn = int(file.readline())
    square = int(file.readline())
    step = int(file.readline())
    file.close()
    return trdata, tsdata, nn, square, step


def getNeighbors(line):
    nbrs = NearestNeighbors(n_neighbors=neighbours, algorithm='auto', metric='cosine').fit(train[:,3:8]) #uses train data without coords and imgnum
    sample = np.asarray(line)
    sample = sample.reshape(1, -1)
    distances, indices = nbrs.kneighbors(sample)
    indices = indices.flatten()
    return indices

def getResponse(indices, labels):
    i = 0
    for x in range(neighbours):
        k = indices[x]
        i += labels[k]
    i = i/neighbours
    return round(i), i

def getInstance(img, meanR, meanG, meanB, mask):
    rB = img[:,:,0].mean() - meanR      #only mean color values of R,G,B (relative to image mean)
    gB = img[:,:,1].mean() - meanG
    bB = img[:,:,2].mean() - meanB
    r = img[:, :, 0][mask].mean() - meanR       #color means inside circle mask
    bbgg = img[:, :, 1:].sum(axis=2)  # leaves only values of blue and green
    gb = bbgg[mask].mean() - meanG - meanB
    rOut = img[:, :, 0][~mask].mean() - meanR       #color means outside circle mask
    bgOut = bbgg[~mask].mean() - meanG - meanB
    sumCircle = r + gb - meanR - meanG - meanB
    #return (rB, gB, bB)
    return (r, gb, rOut, bgOut, sumCircle)

def searchImage(data, arr):
    h, w = arr.shape[:2]
    mask = LoadImage.createCircularMask(h, w)
    arr[:, :, 0][~mask] = 0
    arr[:, :, 1][~mask] = 0
    arr[:, :, 2][~mask] = 0

    meanR = arr[:, :, 0][mask].mean()
    meanG = arr[:, :, 1][mask].mean()
    meanB = arr[:, :, 2][mask].mean()
    meanGB = meanG + meanB

    colonies = []
    results2 = pandas.DataFrame()
    prob_map = np.zeros(shape=(int(h/step),int(w/step)))
    smallMask = LoadImage.createCircularMask(window, window)
    r = int(window/2)       #17/2 = 8

    for i in range(r, arr.shape[0]-r, step):
        for j in range(r, arr.shape[1]-r, step):
            square = arr[int(i - r): int(i + r+1), int(j - r): int(j + r+1), :]
            if np.sum(square) > 0:
               # if((j > 880) and (i > 470)):
               #     a=5
               # if square.shape[1] < window:
               #     i=3
                instance = getInstance(square, meanR, meanG, meanB, smallMask)
                neighbours = getNeighbors(instance)
                response, prob = getResponse(neighbours, y_train)
                if response == 1:
                    colonies.append((i, j, prob))
                results2.loc[i,j] = response
                prob_map[int(i/step), int(j/step)] = prob     #does this indexing work????
    return colonies, results2, prob_map       #changed to results2

def displayColonies(filename, list, root):
    #canvas = Utils.CreateCanvasWImage(filename, root)
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E + W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N + S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N + S + E + W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH, expand=1)

    background_image = PhotoImage(file=filename)
    image = canvas.create_image(0, 0, anchor=NW, image=background_image)
    canvas.config(scrollregion=canvas.bbox(ALL))
    root.wm_geometry("794x370")
    root.title('Map')

    for colony in list:
        x = int(colony[0])
        y = int(colony[1])
        prob = float(colony[2])

        Utils.DisplayPoint(x*ratio, y*ratio, prob, canvas)  #added scaling by ratio
    root.mainloop()

def TestVector(results, testx, testy, num):
    i=1

def Test(results, testx, testy, num, file):
    numC = input('Enter number of colonies.')
    cc = int(numC)
    numOfInst = 0
    acc = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    image = Image.open(file)
    draw = ImageDraw.Draw(image)
    #draw, image = Utils.SimpleImage(file)
    rat = 1/ratio
    for i in range(testx.shape[0]):
#        if i>182:
#            a=8
        if (num == testx[i,10]):
            point = findNearestII(testx[i,9]*rat, testx[i, 8]*rat, results)
            numOfInst += 1
            if (point == testy[i]):
                acc += 1
                if (point == 1):
                    tp+=1
                    #Utils.SimpleDraw(testx[i,8], testx[i,9],"Green",draw)
                    x = testx[i,8]
                    y = testx[i,9]
                    color = "Green"
                    draw.line((x - 4, y - 4, x + 4, y + 4), fill=color, width=2)
                    draw.line((x - 4, y + 4, x + 4, y - 4), fill=color, width=2)
                else:
                    tn+=1
            else:
                if (point == 1):
                    fp+=1
                    #Utils.SimpleDraw(testx[i,8], testx[i,9],"Purple",draw)
                    x = testx[i, 8]
                    y = testx[i, 9]
                    color = "Purple"
                    draw.line((x - 4, y - 4, x + 4, y + 4), fill=color, width=2)
                    draw.line((x - 4, y + 4, x + 4, y - 4), fill=color, width=2)
                else:
                    fn+=1
                    #Utils.SimpleDraw(testx[i, 8], testx[i, 9], "Gray", draw)
                    x = testx[i, 8]
                    y = testx[i, 9]
                    color = "Gray"
                    draw.line((x - 4, y - 4, x + 4, y + 4), fill=color, width=2)
                    draw.line((x - 4, y + 4, x + 4, y - 4), fill=color, width=2)
    image.show()
    print("accuracy:")
    if numOfInst != 0:
        print( acc / numOfInst)
    else:
        print(0)
    print("precision:")
    print(tp/(tp+fp))
    print("recall:")
    print(tp/(tp+fn))

    n = len(col)
    print(cc, 'colonies,', n, ' found.')

    return  acc / numOfInst

def findNearestII(x, y, results):
    dist = 20000
    pt = [[x,y]]
    offset = (int(window/2))%step
    X = x - ((x-offset)%step)       #first square is 8 points from the edge, 8%5=3
    Y = y - ((y-offset)%step)
    XX = X + step
    YY = Y + step
    candidates = []
    candidates.append((X,Y))
    candidates.append((XX, Y))
    candidates.append((X, YY))
    candidates.append((XX, YY))
    for can in candidates:
        new = sklearn.metrics.pairwise_distances([can], pt)[0][0]
        #new = euclideanDistance(x, y, can[0], can[1])
        if (new < dist):
            dist = new
            point = results.loc[can[0], can[1]]
    return point

def euclideanDistance(x1, y1, x2, y2):
    distance = 0
    distance += pow((x1 - x2), 2)
    distance += pow((y1 - y2), 2)
    return math.sqrt(distance)



root = Tk()
#train, test = ChooseInputData(root)
train, test, neighbours, window, step = OpenData()
File = askopenfilename(parent=root, initialdir="M:/", title='Choose an image.')
img2 = Image.open(File)
origWidth = float(img2.size[0])
ratio = origWidth/900
img2 = img2.resize((900,900), Image.ANTIALIAS)
arr = np.array(img2)

y_train = train[:,11]
col, res, prob = searchImage(train, arr)
probmap = pb.ProbMap(prob, window/2, step)
maxmap = probmap.FindMax()
k, cols = probmap.MaxMarking(maxmap)

#with open("col.csv", 'a') as f:
#    writer = csv.writer(f)
#    for line in col:
#        writer.writerow(line)
#res = pandas.read_csv("results.csv")

#res.to_csv("results.csv", header=False)

#with open('col.csv', 'rU') as f:
#    reader = csv.reader(f)
 #   col = list(reader)

displayColonies(File, cols, root)
print(k)
#displayColonies(File, col, arr.shape[0], arr.shape[1], root)
#Test(res, test[:,:11], test[:,11], 6, File)
