from __future__ import print_function
import keras
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,model_from_json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# input image dimensions
img_rows, img_cols = 128, 128

learning_rate = 0.001
b1 = 0.9
b2 = 0.999


#number of output classes
nb_classes = 5
output_class = ['Plot','Table','Histogram', 'Graph', 'Diagram']


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=b1, beta_2=b2, epsilon=None), metrics=['accuracy'])

path_test = '/home/mayank/Desktop/BTP/data'
file = 'test.png'

im = Image.open(path_test+ '/' + file)  
img = im.resize((256,256))
img = img.convert('L')          
file_name = file.split('.')
img.save(path_test +'/' +  file_name[0] + '.jpg', "JPEG")
 
test_imgmatrix = np.array(Image.open(path_test +'/' +  file_name[0] + '.jpg')).flatten()

test_imgmatrix = np.stack([test_imgmatrix]*3, axis=-1)

test_imgmatrix = test_imgmatrix.reshape(1, 256, 256 , 3)

test_imgmatrix = test_imgmatrix.astype('float32')

mean1 = np.mean(test_imgmatrix) # for finding the mean for centering  to zero
test_imgmatrix -= mean1

classes = loaded_model.predict(test_imgmatrix)

print(classes)