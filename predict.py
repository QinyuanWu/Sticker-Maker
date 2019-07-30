from PIL import Image, ExifTags
import dlib
import random
from PIL import Image, ImageDraw, ImageFont
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import argparse
import imutils
import pickle
import cv2
import os
from skimage import io
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import pandas as pd

#helper methods
def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def xpos (charlen):
    ans=(500-fsize(charlen)*charlen)/2
    return ans
def fsize (charlen):
    if charlen<=9:
        return 60
    else:
        ans=500/(charlen+1)
        return int(ans)

def processimg(image):
    image=image.resize((48,48))
    image=image.convert('L')
    image=np.array(image)
    image=image.astype(np.float32)
    image=image.reshape((1,48,48,1))
    return image



# Load image
# img_path = 'test.jpg' ##########################
# image = io.imread(img_path)

#append cropped faces into list
# img = Image.open("test.jpg") ######need to change


#load model
# model = load_model('model.h5')

class Predictor:

    def __init__(self):
        # load model, default graph, and class index
        self.model = load_model('model.h5')
        # self.graph = tf.get_default_graph()
        # self.lb = pickle.loads(open('model/lb.pickle', 'rb').read())

    def predict(self, request):
        f = request.files['image'].read()
        image = cv2.imdecode(np.fromstring(f, np.uint8), cv2.IMREAD_COLOR)
        # print('immmmmmmmmmmmpringting')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # print('immmmmmmmmmmmpringting')
        pilimage = Image.fromarray(image)

        #rotate image
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            exif=dict(image._getexif().items())
            if exif[orientation] == 3:
                image=image.rotate(180, expand=True)
            elif exif[orientation] == 6:
             image=image.rotate(270, expand=True)
            elif exif[orientation] == 8:
              image=image.rotate(90, expand=True)
    

        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass
        uncropped_img=pilimage
        # Detect faces and return crop locations
        detected_faces = detect_faces(image)

        #create a list of cropped images
        images = []
        for x in range(len(detected_faces)):
            newimg = uncropped_img.crop(detected_faces[x])
            images.append(newimg)

        #read mood.csv and create dictionary
        words=pd.read_csv('words.csv')
        words.dropna(inplace=True)
        angry=np.array(words['0'])
        disgust=np.array(words['1'])
        fear=np.array(words['2'])
        happy=np.array(words['3'])
        sad=np.array(words['4'])
        surprise=np.array(words['5'])
        neutral=np.array(words['6'])
        mooddict={0:angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral}

        #output image into current folder, create list of path as return
        num=random.randint(1,101)
        pathlist=[]
        for image in images:
            img = image
            img = img.resize((500,500))
            draw = ImageDraw.Draw(img)
            prob=(self.model.predict(processimg(img)))
            index=np.argmax(prob)
            print(index)
            msg=np.random.choice(mooddict[index])
            msg
            print(msg)
            font = ImageFont.truetype('./STHeiti Medium.ttc', size=fsize(len(str(msg))))
            color = 'rgb(255, 255, 255)' # white coloßr
            draw.text((xpos(len(str(msg))), 400), str(msg), fill=color, font=font)
            img.save('static/output{}.jpg'.format(num)) 
            pathlist.append('output{}.jpg'.format(num))
            num=random.randint(1,101)
        return pathlist
        



