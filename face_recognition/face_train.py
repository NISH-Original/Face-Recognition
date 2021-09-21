import cv2
import os
import numpy as np
from PIL import Image
import pickle
import tkinter as tk

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # getting directory of this file
image_dir = os.path.join(BASE_DIR)

face_cascade = cv2.CascadeClassifier(f'{image_dir}/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {} # getting labels for multiple persons
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		print(file)
		if file.endswith(".png") or file.endswith(".jpg"):   # getting all .png and .jpg files for training
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower() # changing all space bars in names with hyphens
			print(label)
			
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id = label_ids[label]
			
			pil_image = Image.open(path).convert("L") # turning image to grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id)

with open(f"{image_dir}/face-labels.dat", 'wb') as f: # saving the labels dictionary temporarily in pickle file
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels)) # training the data
recognizer.save(f"{image_dir}/face-trainer.yml")

os.system(f'{image_dir}/face_recognize.py')