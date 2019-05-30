from os import listdir
from os.path import isfile, join
import numpy as np
import cv2 as cv

profile_dir='kate'

faces_dir='faces'

profile_imgs = [f for f in listdir(profile_dir) if isfile(join(profile_dir, f))]

face_cascade = cv.CascadeClassifier(join('haarcascades','haarcascade_frontalface_default.xml'))

def detect_face(f):
    print("Detecting face in '%s'..." % f)
    img = cv.imread(join(profile_dir,f))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #It is assumed that the first face detected is the main and the only one in the image
    if(len(faces)>0):
        (x,y,w,h)=faces[0]
        roi = img[y:y+h, x:x+w] 
        cv.imwrite(join(faces_dir,f),roi)
        print("Face written for '%s'." % f)

for profile_img in profile_imgs:
    detect_face(profile_img)
