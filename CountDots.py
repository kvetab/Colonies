import  numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import csv

#opens images, gets examples from clicks on the image
#left click for positive, right click for negative examples, middle button for undo
#enter filename manually...
def openImage():
    event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))
    # if __name__ == "__main__":
    if (True):
        root = Tk()

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
        File = askopenfilename(parent=root, initialdir="M:/", title='Choose an image.')
        print("opening %s" % File)
        img = PhotoImage(file=File)
        img2 = Image.open(File)
        arr = np.array(img2)
        # arr = arr.astype(int)

#        num = input('Enter image number.')
#        imgnum = int(num)
#        posneg = input('Positive or negative examples? 1/0')
#        example = int(posneg)
        coords = []

        print(arr.shape)
        h, w = arr.shape[:2]

        canvas.create_image(0, 0, image=img, anchor="nw")
        canvas.config(scrollregion=canvas.bbox(ALL))

        # function to be called when mouse is clicked
        def printcoordsPos(event):
            # outputting x and y coords to console
            global  colonies
            colonies += 1
            cx, cy = event2canvas(event, canvas)
            canvas.create_line(cx - 4, cy - 4, cx + 4, cy + 4, fill="#2E8B57", width=3)
            canvas.create_line(cx - 4, cy + 4, cx + 4, cy - 4, fill="#2E8B57", width=3)
            coords.append((cx, cy, 1))

        def printcoordsNeg(event):
            cx, cy = event2canvas(event, canvas)
            canvas.create_line(cx - 4, cy - 4, cx + 4, cy + 4, fill="#ed9121", width=3)
            canvas.create_line(cx - 4, cy + 4, cx + 4, cy - 4, fill="#ed9121", width=3)
            coords.append((cx, cy, 0))

        def Undo(Event=None):
            xy = coords.pop()
            x = xy[0]
            y = xy[1]
            canvas.create_oval(x-2, y-2, x+2, y+2, outline="#9400D3", fill="#9400D3")

        # mouseclick event
        canvas.bind("<ButtonPress-1>", printcoordsPos)
        canvas.bind("<ButtonPress-3>",printcoordsNeg)
        canvas.bind_all('<ButtonPress-2>', Undo)

    root.mainloop()

    print(colonies)
    return coords

def SaveToFile(col, filename):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        for line in col:
            writer.writerow(line)

colonies = 0
coords = openImage()
SaveToFile(coords, "coords9570.csv")
