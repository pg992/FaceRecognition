from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
from os import listdir,path, remove
from os.path import isfile, join, isdir, splitext
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
# import matplotlib.pyplot as plt
import cv2 as cv
import boto3
# import sounddevice as sd
import time
from threading import Thread
import calendar;
import time;

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights('vgg_face_weights.h5')

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_loaded_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, target_representation):
    a = np.matmul(np.transpose(source_representation), target_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(target_representation, target_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, target_representation):
    euclidean_distance = source_representation - target_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

face_cascade = cv.CascadeClassifier(join('haarcascades','haarcascade_frontalface_default.xml'))

faces_dir='faces'

faces={}
mainDict={}
dirDict={}
dirs = [f for f in listdir(faces_dir) if isdir(join(faces_dir, f))]
face_imgs=None

def updateDir():
    global face_imgs
    global dirDict 
    for dir in dirs:
        dirDict[dir]=[]
        if face_imgs is None:
            face_imgs = [f for f in listdir(faces_dir + "/" + dir) if isfile(join(faces_dir + "/" + dir, f))]
            for img in face_imgs:
                mainDict[splitext(img)[0]]=dir
                dirDict[dir].append(splitext(img)[0])
        else:
            items=[f for f in listdir(faces_dir + "/" + dir) if isfile(join(faces_dir + "/" + dir, f))]
            for item in items:
                face_imgs.append(item)
                mainDict[splitext(item)[0]]=dir
                dirDict[dir].append(splitext(item)[0])

def updateDirectory(directory):
    global face_imgs
    global dirDict 
    items=[f for f in listdir(faces_dir + "/" + directory) if isfile(join(faces_dir + "/" + directory, f))]
    for item in items:
        if splitext(item)[0] not in dirDict[directory]:
            dirDict[directory].append(splitext(item)[0])
            mainDict[splitext(item)[0]]=directory
            break

updateDir()

# face_imgs = [f for f in listdir(faces_dir) if isfile(join(faces_dir, f))]
def init_faces():
    global face_imgs
    global faces_dir
    global mainDict
    global faces
    for face_file in face_imgs:
        face_label=path.splitext(face_file)[0]
        print(face_label)
        face_representation= vgg_face_descriptor.predict(preprocess_image(join(faces_dir + "/" + mainDict[face_label],face_file)))[0,:]
        faces[face_label]=face_representation

def init_one_face(face_file):
    global face_imgs
    global faces_dir
    global mainDict
    global faces
    face_label=path.splitext(face_file)[0]
    print(face_label)
    # for imgs in dirDict[mainDict[face_label]]:
    if len(dirDict[mainDict[face_label]]) > 5:
        remove(faces_dir + "/" + mainDict[face_label] + "/" + dirDict[mainDict[face_label]][0] + ".jpg")
        dirDict[mainDict[face_label]].pop(0)
    face_representation= vgg_face_descriptor.predict(preprocess_image(join(faces_dir + "/" + mainDict[face_label],face_file)))[0,:]
    faces[face_label]=face_representation

init_faces()

def detect_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces)>0):
        (x,y,w,h)=faces[0]
        roi = img[y:y+h, x:x+w] 
        return roi


while True:
    try:
        vc = cv.VideoCapture(0)
        if vc.isOpened(): 
            is_capturing, frame = vc.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            vc.release()
            face=detect_face(frame)
            # cv.imwrite(faces_dir +"frame1.jpg", face)
            # plt.imshow(face)
            face=cv.resize(face,(224,224))
            face = face[...,::-1]
            face_representation= vgg_face_descriptor.predict(preprocess_loaded_image(face))[0,:]
            min_sim=0.4
            candidate=''
            finalKey=''
            for key in faces.keys():
                candidate_representation=faces[key]
                cosine_similarity = findCosineSimilarity(face_representation, candidate_representation) # Should be less then 0.40
                euclidean_distance = findEuclideanDistance(face_representation, candidate_representation) #Less then 120
                # print("Candidate {} CosineSimularity: {}, EuclideanDistance: {}, Key: {}" .format(mainDict[key], cosine_similarity, euclidean_distance, key))
                if cosine_similarity<min_sim:
                    min_sim=cosine_similarity
                    candidate=mainDict[key]
                    finalKey=key
            ts = calendar.timegm(time.gmtime())        
            cv.imwrite(faces_dir + "/" + candidate + "/" + candidate + "-" + str(ts) + ".jpg", face)
            updateDirectory(candidate)
            init_one_face(candidate + "-" + str(ts) + ".jpg")
            print("Candidate {}, Key {}, Sim {}".format(candidate, finalKey, min_sim))
            time.sleep(2)
    except Exception as ex:
        print("No face")
        time.sleep(2)
        continue
    
    # speak('Hello '+candidate+'. May I help you?')


# def speak(text):
#     response = polly_client.synthesize_speech(VoiceId='Brian',OutputFormat='pcm',SampleRate="8000",Text = text)
#     stream=response['AudioStream'].read()
#     sound=np.frombuffer(stream,dtype=np.int16)
#     sd.play(sound, 8000)