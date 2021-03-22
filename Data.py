import numpy as np
import os
import cv2
import random
import pickle

dataDir = "C:/Users/Bradley Blyther/Documents/Datasets/PetImages"
category = ["Dog", "Cat"]
imgSize = 100
trainingData = []

def createTrainingData():
    for c in category:
        path = os.path.join(dataDir, c)
        classNum = category.index(c)
        for img in os.listdir(path):
            try:
                imgArray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                newArray = cv2.resize(imgArray, (imgSize, imgSize))
                trainingData.append([newArray, classNum])
            except Exception as e:
                pass

createTrainingData()
random.shuffle(trainingData)

x = []
y = []

for features, label in trainingData:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, imgSize, imgSize, 1)

pickleOut = open("x.pickle", "wb")
pickle.dump(x, pickleOut)
pickleOut.close()

pickleOut = open("y.pickle", "wb")
pickle.dump(y, pickleOut)
pickleOut.close()