import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import datetime
import os
from tkinter import messagebox
import sys
import os

dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# making main window (for now, atleast)
root = tk.Tk()
root.geometry('700x640')
root.title('Face Recognition Prototype')
root.configure(bg='#3d3d3d')
title = tk.Label(root, text='Face Recognition',font=("Arial", 25, 'bold'), bg='#3d3d3d', fg='white')
title.place(relx=0.5,rely=0.05,anchor=tk.CENTER)
f1 = tk.LabelFrame(root, bg='#3d3d3d')
f1.place(relx=0.5, rely=0.53, anchor=tk.CENTER)
feed = tk.Label(f1)
feed.pack()
steps_label = tk.Label(root, text = "Hello! First please enter your name in the textbox,\nthen press the 'enter' button ->", bg='#3d3d3d', fg='light green', font=('Arial', 12))
steps_label.place(relx=0.5, rely=0.11, anchor=tk.CENTER)

# taking pictures
def TakePic():
    global step_count
    global img1
    global p
    image = Image.fromarray(img1)
    img_name = str(datetime.datetime.now().today()).replace(":","")+".jpg"
    image.save(f'{p}/{img_name}')
    step_count+=1
    steps_label.config(text=f'{steps[step_count-1]}', bg='#3d3d3d', fg='light green', font=('Arial', 12))

cap_button = tk.Button(root, text='📷 Take Picture', font=('Arial',15), bg='#3d3d3d',fg='white', command=TakePic)
cap_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
cap_button.config(state=tk.DISABLED)

# entry for getting name
name_entry = tk.Entry(root)
name_entry.place(relx=0.76, rely=0.097)
def getname(event=None):    
    global p
    name = name_entry.get()
    d = os.path.dirname(__file__) # directory of script
    p = f'{d}/{name}' # path to be created
    if not os.path.exists(name):
        os.makedirs(p)
        cap_button.config(state=tk.NORMAL)
        name_entry.destroy()
        steps_label.config(text="Keep your face STRAIGHT and take a picture")
    else:
        messagebox.showerror("ERROR", "A folder with the same name already exists!")
name_entry.bind('<Return>', getname)

steps = [
        "Look a bit to the RIGHT and take a picture",
        "Look a bit to the LEFT and take a picture",
        "Tilt your face a bit to the RIGHT and take a picture",
        "Tilt your face a bit to the LEFT and take a picture",
        "Look a bit DOWN and take a picture",
        "Look a bit UP and take a picture",
        "Perfect! All your pictures are saved :)\nTraining will start in a moment"
    ]

global step_count
step_count = 0

# this is the code which makes the computer get webcam feed
cap = cv2.VideoCapture(0)

def start_next_code():
    os.system(f'{dir}/face_train.py')
    try:
        root.after(2000,lambda:root.destroy())
    except:
        pass

def close():
    cap.release()
    start_next_code()

# the camera will also close when the window is closed
root.protocol("WM_DELETE_WINDOW", close)

def capture():
    if cap.isOpened():
        ret, img = cap.read()
        if ret:
            img = cv2.flip(img, 1)
            global img1
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # changing color to RGB
            img = ImageTk.PhotoImage(Image.fromarray(img1))
            feed['image'] = img  # putting the webcam feed in the 'feed' LabelFrame
            feed.image = img # save reference of the image

            if step_count >= 7:
                cap_button.config(state=tk.DISABLED)
                cap.release()
                start_next_code()
    
    root.after(10, capture)

capture()

root.mainloop()



