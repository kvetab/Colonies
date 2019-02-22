from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image
from PIL import ImageDraw
import math

#enter test data and list of nearest found points and display these in the photo
def CreateCanvasWImage(filename, root):
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

    return canvas

def DisplayPoint(x,y,prob, canvas):
    color = str(int(min((200 - round(prob * 200), 99))))
    canvas.create_arc(y - 8, x - 8, y + 8, x + 8, outline="#" + color * 3, width=2, extent=360)


def SimpleImage(filename):
    image = Image.open(filename)
    #image.show()
    draw = ImageDraw.Draw(image)
    return draw, image

def SimpleDraw(x,y,color,draw):
    draw.line((x-4, y-4, x+4, y+4), fill=color, width=2)
    draw.line((x - 4, y + 4, x + 4, y - 4), fill=color, width=2)


def SaveSettings(step, win, train, test):
    f = open("settings.txt", 'w')
    f.write(step)
    f.write(win)
    f.write(train)
    f.write(test)
    f.close()

def getValues(img, meanR, meanG, meanB, mask, x, y, num, TF):
    rB = img[:,:,0].mean() - meanR      #only mean color values of R,G,B (relative to image mean)
    gB = img[:,:,1].mean() - meanG
    bB = img[:,:,2].mean() - meanB
    r = img[:, :, 0][mask].mean() - meanR       #color means inside circle mask
    bbgg = img[:, :, 1:].sum(axis=2)  # leaves only values of blue and green
    gb = bbgg[mask].mean() - meanG - meanB
    rOut = img[:, :, 0][~mask].mean() - meanR       #color means outside circle mask
    bgOut = bbgg[~mask].mean() - meanG - meanB
    sumCircle = r + gb - meanR - meanG - meanB

    return [rB, gB, bB, r, gb, rOut, bgOut, sumCircle, x, y, num, TF]