import numpy as np
from tqdm import tqdm
import cv2
import pickle
import os

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

picture_folder = ".."
target_folder = ".."
all_images = os.listdir(picture_folder)

for item in tqdm(all_images) :
    img = cv2.imread(picture_folder + item)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 1)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        cv2.imwrite(target_folder + item, roi_color)
