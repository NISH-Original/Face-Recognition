import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import datetime
import os
import pickle

global dir
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
steps_label = tk.Label(root, text = 'Now lets see if I can recognize you!', bg='#3d3d3d', fg='light green', font=('Arial', 12))
steps_label.place(relx=0.5, rely=0.11, anchor=tk.CENTER)

face_cascade = cv2.CascadeClassifier(f'{dir}/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(f'{dir}/face-trainer.yml')

old_labels = {}
with open(f'{dir}/face-labels.dat', 'rb') as pf:
    old_labels = pickle.load(pf)
    labels = {value:key for key, value in old_labels.items()}

# this is the code which makes the computer get webcam feed
cap = cv2.VideoCapture(0)

def close():
    cap.release()
    cv2.destroyAllWindows()

# the camera will also close when the window is closed
root.protocol("WM_DELETE_WINDOW", close)

def close():
    global dir
    cap.release()
    os.remove(f'{dir}/face-labels.dat')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # getting directory of this file
    dir = os.path.join(BASE_DIR)

    for root, dirs, files in os.walk(dir):
        for dir in dirs:
            print(dir)
            if dir.endswith(".png") or dir.endswith(".jpg"): 
                pass

# the camera will also close when the window is closed
root.protocol("WM_DELETE_WINDOW", close)

while cap.isOpened():
    global img1
    img = cap.read()[1]  
    img = cv2.flip(img, 1)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # changing color to RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img1, scaleFactor=1.3,minNeighbors=5)
    for x,y,w,h in faces:
        roi = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        # recognising the person in video
        id, conf = recognizer.predict(roi_gray) # conf is amt of confidence of prediction
        if conf >= 40:
            name = labels[id] 
        
        img = cv2.rectangle(img1, (x,y), (x+w,y+h), (255,0,0), 3)
        cv2.putText(img,f'Name: {name.capitalize()}',(x,y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        cv2.putText(img,f'Confidence: {round(conf, 2)}',(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    
    img = ImageTk.PhotoImage(Image.fromarray(img1))
    feed['image'] = img  # putting the webcam feed in the 'feed' LabelFrame

    root.update()



