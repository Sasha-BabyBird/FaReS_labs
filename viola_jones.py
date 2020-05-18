import numpy as np
import cv2
import matplotlib.pyplot as plt
import os



def detect_face(image_path, face_cascade_path, eye_cascade_path):
    '''
    '''
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    img_color = cv2.imread(image_path)
    if len(img_color.shape) == 2:
        img_gray = img_color
    else:
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.01, 
            minNeighbors=50, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        img_color = cv2.rectangle(img_color, (x, y),(x+w, y+h), (255, 0, 0), 2)

        #ROI = Region of Interest
        roi_gray = img_gray[y:y+h, x:x+w]
        roi_color = img_color[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.01, 
                minNeighbors=25, minSize=(1, 1))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    cv2.imshow('img', img_color)
    cv2.waitKey(0)

if __name__ == '__main__':
    photos_dir = os.path.join('.', 'photos') 
    #my_photo = os.path.join(photos_dir, 'photo_8.jpg')
    my_photo = os.path.join('.', 'ORL Face Database', 's7', '3.pgm')
    face_cascade_path = os.path.join('.', 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join('.', 'haarcascade_eye.xml')
    detect_face(my_photo, face_cascade_path, eye_cascade_path)


