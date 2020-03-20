from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory
from Predictor import PredictorKeras
from PIL import Image, ImageTk
import os


DEFAULT_MODEL = "models/model1584474209.720228"

window = Tk()
window.title("ColonyCount")
window.geometry('350x500')
# MODEL = PredictorKeras(DEFAULT_MODEL)
MODEL_DIR = DEFAULT_MODEL

def choose_model(event):
    global MODEL, MODEL_DIR
    model_dir = askdirectory(initialdir=os.getcwd()+"/models", title='Choose a model directory.', mustexist=True)
    if model_dir:
        MODEL = PredictorKeras(model_dir)
        MODEL_DIR = model_dir

def choose_photo(event):
    img_file = askopenfilename(initialdir=os.path.join(os.getcwd(), "photos_used"), title='Choose an image.')
    win2 = Toplevel(window)
    img = Image.open('photos_used/PICT9575.png')



    cv = Canvas(win2)
    photo = ImageTk.PhotoImage(img)
    cv.grid(column=0, row=0)
    cv.create_image(50, 50, image=photo, anchor='nw')
    win2.mainloop()

    """
    img_file = askopenfilename(initialdir=os.path.join(os.getcwd(), "photos_used"), title='Choose an image.')
    win2 = Toplevel(window)
    cv = Canvas(win2)
    #frame.pack(fill=BOTH, expand=1)
    #img_file = 'photos_used/PICT9575.png'
    img = Image.open(img_file)
    photo = ImageTk.PhotoImage(img_file)
    cv.grid(column=0, row=3)
    cv.create_image(50, 50, image=photo, anchor='nw')
    """


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

