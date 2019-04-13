#ALL IMPORTS
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
from keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# # input image dimensions
# img_rows, img_cols = 128, 128

# # number of channels
# img_channels = 1
# # number of epochs
# epochs=20

# # number of output classes
# nb_classes = 3

# #  data
# path1_table = '/home/mayank/Desktop/BTP/data/table'    #path of folder of table images    
# path2_table_processed ='/home/mayank/Desktop/BTP/data/table_processed'  #path of folder to save table images

# path1_plot = '/home/mayank/Desktop/BTP/data/plot'    #path of folder of plots images
# path2_plot_processed ='/home/mayank/Desktop/BTP/data/plot_processed'  #path of folder to save plots images

# path1_diagram = '/home/mayank/Desktop/BTP/data/diagram'    #path of folder of diagram images
# path2_diagram_processed ='/home/mayank/Desktop/BTP/data/diagram_processed'  #path of folder to save diagram images

# list_table = os.listdir(path1_table)   #Reading all the images from table folder
# list_plot = os.listdir(path1_plot)     #Reading all the images from plot folder
# list_diagram = os.listdir(path1_diagram) #Reading all the images from diagram folder

# table_samples = size(list_table)        #Getting the total number of Images in Table folder
# plot_samples = size(list_plot)          #Getting the total number of Images in plot folder
# diagram_samples = size(list_diagram)    #Getting the total number of Images in Diagram folder

# num_samples = table_samples + plot_samples + diagram_samples #Total number of Images
# print(num_samples)

# # save the processed table images
# for file in list_table:
#     im = Image.open(path1_table + '/' + file)  
#     img = im.resize((img_rows,img_cols))
#     img = img.convert('RGB')          
#     img.save(path2_table_processed +'/' +  file, "JPEG")


# # save the processed plot images
# for file in list_plot:
#     im = Image.open(path1_plot + '/' + file)  
#     img = im.resize((img_rows,img_cols))
#     img = img.convert('RGB')         
#     img.save(path2_plot_processed +'/' +  file, "JPEG")


# # save the processed diagram images
# for file in list_diagram:
#     im = Image.open(path1_diagram + '/' + file)  
#     img = im.resize((img_rows,img_cols))
#     img = img.convert('RGB')          
#     img.save(path2_diagram_processed +'/' +  file, "JPEG")


# # img_list = os.listdir(path2)
# # img1 = array(Image.open(path2 +'/'+ img_list[0])) # open one image to get size
# # m,n = img1.shape[0:2] # get the size of the images
# # # print(m,n)
# # img_number = size(img_list) # get the number of images
# # # print(img_number)


# #Processing for table images
# table_imglist = os.listdir(path2_table_processed)
# print (table_imglist)
# table_img1 = array(Image.open(path2_table_processed + '/'+ table_imglist[0])) # open one table image to get size
# table_m,table_n = table_im1.shape[0:2] # get the size of the table images
# table_image_number = len(table_imglist) # get the number of table images

# # create matrix to store all flattened table images
# table_imgmatrix = array([array(Image.open(path2_table_processed + '/' + table_img2)).flatten()
#               for table_img2 in table_imglist],'f')



# #Processing for plot images
# plot_imglist = os.listdir(path2_plot_processed)
# print (plot_imglist)
# plot_img1 = array(Image.open(path2_plot_processed + '/'+ plot_imglist[0])) # open one plot image to get size
# plot_m,plot_n = plot_img1.shape[0:2] # get the size of the plot images
# plot_image_number = len(plot_imglist) # get the number of plot images

# # create matrix to store all flattened plot images
# plot_imgmatrix = array([array(Image.open(path2_plot_processed + '/' + plot_img2)).flatten()
#               for plot_img2 in plot_imglist],'f')


# #Processing for diagram images
# diagram_imglist = os.listdir(path2_diagram_processed)
# print (diagram_imglist)
# diagram_img1 = array(Image.open(path2_diagram_processed + '/'+ diagram_imglist[0])) # open one diagram image to get size
# diagram_m,diagram_n = diagram_img1.shape[0:2] # get the size of the diagram images
# diagram_image_number = len(diagram_imglist) # get the number of diagram images

# # create matrix to store all flattened diagram images
# diagram_imgmatrix = array([array(Image.open(path2_diagram_processed + '/' + diagram_img2)).flatten()
#                for diagram_img2 in diagram_imglist],'f')




# # Combining each matrix
# imgmatrix = numpy.concatenate((table_imgmatrix, plot_imgmatrix), axis=0)
# imgmatrix = numpy.concatenate((imgmatrix, diagram_imgmatrix), axis=0)


# # label=numpy.ones((num_samples,),dtype = int)
# # label[0:101]=0
# # label[102:201]=1
# # label[202:]=2

# data,Label = shuffle(imgmatrix,label, random_state=2)
# train_data = [data,Label]

# (X, y) = (train_data[0],train_data[1])

# #split X and y into training and testing sets

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

# # Save the test images in a file for displaying
# size = X_test.shape[0]
# for imgn in range(size):
#     img = Image.fromarray(X_test[imgn][0])
#     img = img.convert('RGB')
#     img.save('/home/mayank/Desktop/BTP/data/test_image' +'/' +  str(imgn), "JPEG")


# # Data preprocessing
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# mean1 = numpy.mean(X_train) # for finding the mean for centering  to zero
# X_train -= mean1
# X_test -= mean1


# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

nb_classes = 3

model = ResNet50(include_top =False, weights='imagenet', input_shape=(128,128,3), pooling='max')
model.summary()

x = model.output

# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(nb_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
resnet_model2 = Model(inputs=model.input, outputs=out)

resnet_model2.summary()

# for layer in resnet_model2.layers[:-6]:
# 	layer.trainable = False

#    resnet_model2.layers[-1].trainable

# resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# y_train = y_train.reshape((-1, 1))
# # Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=16, nb_epoch=epochs, verbose=2)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
