from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory
from Predictor import PredictorKeras
from PIL import Image, ImageTk
import os
import CountDots


DEFAULT_MODEL = "models/model1584474209.720228"

window = Tk()
window.title("ColonyCount")
window.geometry('300x150')
window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=1)
window.rowconfigure(2, weight=1)
window.columnconfigure(1, weight=1)
window.columnconfigure(0, weight=1)

MODEL = PredictorKeras(DEFAULT_MODEL)
MODEL_DIR = DEFAULT_MODEL
colonies = 0

def choose_model(event):
    global MODEL, MODEL_DIR
    model_dir = askdirectory(initialdir=os.getcwd()+"/models", title='Choose a model directory.', mustexist=True)
    if model_dir:
        MODEL = PredictorKeras(model_dir)
        MODEL_DIR = model_dir

def choose_photo(event):
    img_file = askopenfilename(initialdir=os.path.join(os.getcwd(), "photos_used"), title='Choose an image.')
    img = Image.open(img_file)
    win2 = Toplevel(window)
    win2.geometry('600x400')
    win2.rowconfigure(0, weight=2)
    win2.columnconfigure(0, weight=1)
    win2.columnconfigure(1, weight=1)


    frame = Frame(win2, bd=2, relief=SUNKEN)
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
    frame.grid(column=0, row=0, sticky=N+E+W+S, columnspan=2)

    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=photo, anchor='nw')
    canvas.config(scrollregion=canvas.bbox(ALL))

    prediction, positive = MODEL.predict(img_file, verbose=0)
    prediction_label = Label(win2, text="Prediction is: " + str(prediction), justify=LEFT, anchor=W)
    prediction_label.grid(column=0, row=1, sticky=W)

    prediction_label_pos = Label(win2, text="Positive prediction is: " + str(positive), justify=LEFT, anchor=W)
    prediction_label_pos.grid(column=0, row=2, sticky=W)

    def count_with_args(event):
        count_manually(canvas, img_file, win2)

    button_count = Button(win2, text="Count manually", justify=RIGHT, anchor=E)
    button_count.bind("<Button-1>", count_with_args)
    button_count.grid(column=1, row=1, sticky=E)

    win2.mainloop()


def count_manually(canvas, photo, win):
    mark_size = 3
    mark_width = 2
    coords = []
    global colonies
    colonies = 0

    def save(event):
        CountDots.SaveToFile(coords, coords_file, "coords/")
        button_save.destroy()
        event2canvas = None
        canvas.bind("<ButtonPress-1>", do_nothing)
        canvas.bind("<ButtonPress-3>", do_nothing)
        canvas.bind_all('<ButtonPress-2>', do_nothing)

    button_save = Button(win, text="Save", justify=RIGHT, anchor=E)
    button_save.bind("<Button-1>", save)
    button_save.grid(column=1, row=2, sticky=E)

    event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))
    # function to be called when mouse is clicked
    def printcoordsPos(event):
        # outputting x and y coords to console
        global colonies
        colonies += 1
        cx, cy = event2canvas(event, canvas)
        color = "#32b33b"
        canvas.create_line(cx - mark_size, cy - mark_size, cx + mark_size, cy + mark_size, fill=color, width=mark_width)
        canvas.create_line(cx - mark_size, cy + mark_size, cx + mark_size, cy - mark_size, fill=color, width=mark_width)
        coords.append((cx, cy, 1))

    def printcoordsNeg(event):
        cx, cy = event2canvas(event, canvas)
        canvas.create_line(cx - mark_size, cy - mark_size, cx + mark_size, cy + mark_size, fill="#ed9121",
                           width=mark_width)
        canvas.create_line(cx - mark_size, cy + mark_size, cx + mark_size, cy - mark_size, fill="#ed9121",
                           width=mark_width)
        coords.append((cx, cy, 0))

    def Undo(Event=None):
        xy = coords.pop()
        x = xy[0]
        y = xy[1]
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline="#9400D3", fill="#9400D3")

    def do_nothing(event):
        pass

    # mouseclick event
    canvas.bind("<ButtonPress-1>", printcoordsPos)
    canvas.bind("<ButtonPress-3>", Undo)
    canvas.bind_all('<ButtonPress-2>', printcoordsNeg)

    coords_file = os.path.basename(photo).replace("PICT", "coords")
    coords_file = coords_file.replace("png", "csv")
    print(coords_file)
    coords_file = "pokus"


label_model = Label(window, text="Current model: ")
label_model.grid(column=0, row=0)

label_model_number = Label(window, text=MODEL_DIR)
label_model_number.grid(column=1, row=0)

button_model = Button(window, text="Choose model")
button_model.bind("<Button-1>", choose_model)
button_model.grid(column=0, row=1)

button_photo = Button(window, text="Choose photo")
button_photo.bind("<Button-1>", choose_photo)
button_photo.grid(column=0, row=2)



window.mainloop()

