from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory
from Predictor import PredictorKeras
from PIL import Image, ImageTk
import os


DEFAULT_MODEL = "models/model1584474209.720228"

window = Tk()
window.title("ColonyCount")
window.geometry('400x200')
MODEL = PredictorKeras(DEFAULT_MODEL)
MODEL_DIR = DEFAULT_MODEL

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
    frame.pack(fill=BOTH, expand=1)

    #cv = Canvas(win2)
    photo = ImageTk.PhotoImage(img)
    #cv.grid(column=0, row=0)
    canvas.create_image(0, 0, image=photo, anchor='nw')
    canvas.config(scrollregion=canvas.bbox(ALL))

    prediction, positive = MODEL.predict(img_file, verbose=0)
    #prediction_label = Label(win2, text="Prediction is: " + str(prediction))
    prediction_label = Label(win2, text="Prediction is: ")
    prediction_label.grid_rowconfigure(1)
    prediction_label.grid_columnconfigure(0)
    prediction_label.pack()
    #prediction_label_pos = Label(win2, text="Prediction is: " + str(positive))
    prediction_label_pos = Label(win2, text="Prediction is: ")
    prediction_label_pos.grid_rowconfigure(2)
    prediction_label_pos.grid_columnconfigure(0)
    prediction_label_pos.pack()


    win2.mainloop()



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

