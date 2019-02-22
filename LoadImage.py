import  numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import csv

class ImageLoader:
    File=''
    values = []
    def __init__(self, col, emp):
        self.col=col
        self.emp=emp
        self.imgnum = 0
        self.count = 0

    def SaveToFile(self, col):
        if col==True:
            filename = self.col
        else:
            filename = self.emp
        with open(filename, 'a') as f:
            writer = csv.writer(f)
            for line in self.values:
                if col==True:
                    line.append(1)
                else:
                    line.append(0)
                writer.writerow(line)

    def openImage(self):
        event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))
        # if __name__ == "__main__":
        if (True):
            root = Tk()
            self.values = []

            # setting up a tkinter canvas with scrollbars
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

            # adding the image
            self.File = askopenfilename(parent=root, initialdir="M:/", title='Choose an image.')
            print("opening %s" % self.File)
            img = PhotoImage(file=self.File)
            img2 = Image.open(self.File)
            arr = np.array(img2)
            #arr = arr.astype(int)

            num = input('Enter image number.')
            self.imgnum = int(num)

            print(arr.shape)
            h, w = arr.shape[:2]
            mask = createCircularMask(h, w)
            smallMask = createCircularMask(17, 17)

            meanR = arr[:,:,0][mask].mean()
            meanG = arr[:,:,1][mask].mean()
            meanB = arr[:,:,2][mask].mean()
            sumMean = meanB + meanG + meanR
            print(meanR, meanG, meanB)
            canvas.create_image(0, 0, image=img, anchor="nw")
            canvas.config(scrollregion=canvas.bbox(ALL))


            #meanR = arr[:, :, 0].mean()
            #meanGB = arr[:, :, 1:].mean()
            meanGB = (meanG + meanB)
            print(meanR)
            print(self.count)


            # function to be called when mouse is clicked
            def printcoords(event):
                # outputting x and y coords to console
                self.count += 1
                cx, cy = event2canvas(event, canvas)
                print("(%d, %d) / (%d, %d)" % (event.x, event.y, cx, cy))
                print(arr[int(cy), int(cx)])
                square = arr[int(cy - 8): int(cy + 9), int(cx - 8): int(cx + 9), :] # 17*17 square
                canvas.create_line(cx - 4, cy - 4, cx + 4, cy + 4, fill="#2E8B57", width=3)
                canvas.create_line(cx - 4, cy + 4, cx + 4, cy - 4, fill="#2E8B57", width=3)

                self.values.append(getValues(square, meanR, meanG, meanB, smallMask, cx, cy, self.imgnum))



            # mouseclick event
            canvas.bind("<ButtonPress-1>", printcoords)
            # canvas.bind("<ButtonRelease-1>",printcoords)

            root.mainloop()

def createCircularMask(h, w):

    center = [int(w/2), int(h/2)]
    radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def getValues(img, meanR, meanG, meanB, mask, x, y, num):
    rB = img[:,:,0].mean() - meanR      #only mean color values of R,G,B (relative to image mean)
    gB = img[:,:,1].mean() - meanG
    bB = img[:,:,2].mean() - meanB
    r = img[:, :, 0][mask].mean() - meanR       #color means inside circle mask
    bbgg = img[:, :, 1:].sum(axis=2)  # leaves only values of blue and green
    gb = bbgg[mask].mean() - meanG - meanB
    rOut = img[:, :, 0][~mask].mean() - meanR       #color means outside circle mask
    bgOut = bbgg[~mask].mean() - meanG - meanB
    sumCircle = r + gb - meanR - meanG - meanB

    return [rB, gB, bB, r, gb, rOut, bgOut, sumCircle, x, y, num]