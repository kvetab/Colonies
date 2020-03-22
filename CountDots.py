import  numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageDraw
import csv
import CreateData

#opens images, gets examples from clicks on the image
#left click for positive, right click for negative examples, middle button for undo
#enter filename manually...
def openImage(coord_file):
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

        mark_size = 3
        mark_width = 2

        #coord_file = File.replace("PICT", "coords").replace("png", "csv")
        #coord_file = "coords9575.csv"
        try:
            coord_list = CreateData.LoadCoords(coord_file, "coords/")
            for c in coord_list:
                x = c[0]
                y = c[1]
                if c[2] == 1:
                    canvas.create_line(x - mark_size, y - mark_size, x + mark_size, y + mark_size, fill="blue", width=mark_width)
                    canvas.create_line(x - mark_size, y + mark_size, x + mark_size, y - mark_size, fill="blue", width=mark_width)
        except:
            pass

        # function to be called when mouse is clicked
        def printcoordsPos(event):
            # outputting x and y coords to console
            global  colonies
            colonies += 1
            cx, cy = event2canvas(event, canvas)
            color = "#32b33b"
            canvas.create_line(cx - mark_size, cy - mark_size, cx + mark_size, cy + mark_size, fill=color, width=mark_width)
            canvas.create_line(cx - mark_size, cy + mark_size, cx + mark_size, cy - mark_size, fill=color, width=mark_width)
            coords.append((cx, cy, 1))

        def printcoordsNeg(event):
            cx, cy = event2canvas(event, canvas)
            canvas.create_line(cx - mark_size, cy - mark_size, cx + mark_size, cy + mark_size, fill="#ed9121", width=mark_width)
            canvas.create_line(cx - mark_size, cy + mark_size, cx + mark_size, cy - mark_size, fill="#ed9121", width=mark_width)
            coords.append((cx, cy, 0))

        def Undo(Event=None):
            xy = coords.pop()
            x = xy[0]
            y = xy[1]
            canvas.create_oval(x-2, y-2, x+2, y+2, outline="#9400D3", fill="#9400D3")

        # mouseclick event
        canvas.bind("<ButtonPress-1>", printcoordsPos)
        canvas.bind("<ButtonPress-3>",Undo)
        canvas.bind_all('<ButtonPress-2>', printcoordsNeg)

    root.mainloop()

    print(colonies)
    return coords

def SaveToFile(col, filename, dct):
    with open(dct+filename, 'a') as f:
        writer = csv.writer(f)
        for line in col:
            writer.writerow(line)

def ShowCoords(img_file, coord_file):
    # root = Tk()
    # File = askopenfilename(parent=root, initialdir="M:/", title='Choose an image.')
    # print("opening %s" % File)
    # img = PhotoImage(file=File)
    IMG_DCT = "photos_used/"
    img = Image.open(IMG_DCT + img_file)
    draw = ImageDraw.Draw(img)
    COORDS_DCT = "coords/"
    #COORDS_DCT = ""
    coord_list = CreateData.LoadCoords(coord_file, COORDS_DCT)
    for c in coord_list:
        if c[2] == 1:
            x = c[0]
            y = c[1]
            draw.line([(x-4, y-4), (x+4, y+4)], fill="green", width=2)
            draw.line([(x-4, y+4), (x+4, y-4)], fill="green", width=2)
    del draw
    Image._show(img)


if __name__ == "__main__":
    colonies = 0
    filename = "odkladaci.csv"
    coords = openImage( filename)
    SaveToFile(coords, filename, "coords/")
    #ShowCoords("PICT2.png", "coords2.csv")
    #openImage("blah")
