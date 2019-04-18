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
from keras import utils as np_utils
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# input image dimensions
img_rows, img_cols = 256,256

# number of channels
img_channels = 3

# number of epochs
epoch = 50

# number of output classes
nb_classes = 5

#  data
path1_table = '/home/mayank/Desktop/BTP/data/table'    #path of folder of table images    
path2_table_processed ='/home/mayank/Desktop/BTP/data/table_processed'  #path of folder to save table images

path1_plot = '/home/mayank/Desktop/BTP/data/plot'    #path of folder of plots images
path2_plot_processed ='/home/mayank/Desktop/BTP/data/plot_processed'  #path of folder to save plots images

path1_diagram = '/home/mayank/Desktop/BTP/data/diagram'    #path of folder of diagram images
path2_diagram_processed ='/home/mayank/Desktop/BTP/data/diagram_processed'  #path of folder to save diagram images

path1_histogram = '/home/mayank/Desktop/BTP/data/histogram'    #path of folder of histogram images
path2_histogram_processed ='/home/mayank/Desktop/BTP/data/histogram_processed'  #path of folder to save histogram images

path1_graph = '/home/mayank/Desktop/BTP/data/graph'    #path of folder of graph images
path2_graph_processed ='/home/mayank/Desktop/BTP/data/graph_processed'  #path of folder to save graph images


list_table = os.listdir(path1_table)   #Reading all the images from table folder
list_plot = os.listdir(path1_plot)     #Reading all the images from plot folder
list_diagram = os.listdir(path1_diagram) #Reading all the images from diagram folder
list_histogram = os.listdir(path1_histogram) #Reading all the images from histogram folder
list_graph = os.listdir(path1_graph) #Reading all the images from graph folder

table_samples = size(list_table)        #Getting the total number of Images in Table folder
plot_samples = size(list_plot)          #Getting the total number of Images in plot folder
diagram_samples = size(list_diagram)    #Getting the total number of Images in Diagram folder
histogram_samples = size(list_histogram)#Getting the total number of Images in Histogram folder
graph_samples = size(list_graph)        #Getting the total number of Images in Graph folder

num_samples = table_samples + plot_samples + diagram_samples + histogram_samples + graph_samples #Total number of Images
print(num_samples)

# save the processed table images
for file in list_table:
    im = Image.open(path1_table + '/' + file)  
    img = im.resize((img_rows,img_cols))
    img = img.convert('L')          
    file_name = file.split('.')
    img.save(path2_table_processed +'/' +  file_name[0] + '.jpg', "JPEG")

# save the processed plot images
for file in list_plot:
    im = Image.open(path1_plot + '/' + file)  
    img = im.resize((img_rows,img_cols))
    img = img.convert('L')     
    file_name = file.split('.')    
    img.save(path2_plot_processed +'/' + file_name[0] + '.jpg', "JPEG")

# save the processed diagram images
for file in list_diagram:
    im = Image.open(path1_diagram + '/' + file)  
    img = im.resize((img_rows,img_cols))
    img = img.convert('L')     
    file_name = file.split('.')     
    img.save(path2_diagram_processed +'/' +file_name[0] + '.jpg', "JPEG")
 
#save the processed histogram images
for file in list_histogram:
    im = Image.open(path1_histogram + '/' + file)  
    img = im.resize((img_rows,img_cols))
    img = img.convert('L')     
    file_name = file.split('.')     
    img.save(path2_histogram_processed +'/' + file_name[0] + '.jpg', "JPEG")

#save the processed graph images
for file in list_graph:
    im = Image.open(path1_graph + '/' + file)  
    img = im.resize((img_rows,img_cols))
    img = img.convert('L')     
    file_name = file.split('.')     
    img.save(path2_graph_processed +'/' + file_name[0] + '.jpg', "JPEG")


#Processing for table images
table_imglist = os.listdir(path2_table_processed)
print (table_imglist)
table_img1 = array(Image.open(path2_table_processed + '/'+ table_imglist[0])) # open one table image to get size
table_m,table_n = table_img1.shape[0:2] # get the size of the table images
table_image_number = len(table_imglist) # get the number of table images

# create matrix to store all flattened table images
table_imgmatrix = np.array([np.array(Image.open(path2_table_processed + '/' + table_img2)).flatten() for table_img2 in table_imglist])



#Processing for plot images
plot_imglist = os.listdir(path2_plot_processed)
print (plot_imglist)
plot_img1 = array(Image.open(path2_plot_processed + '/'+ plot_imglist[0])) # open one plot image to get size
plot_m,plot_n = plot_img1.shape[0:2] # get the size of the plot images
plot_image_number = len(plot_imglist) # get the number of plot images

# create matrix to store all flattened plot images
plot_imgmatrix = np.array([np.array(Image.open(path2_plot_processed + '/' + plot_img2)).flatten() for plot_img2 in plot_imglist])


#Processing for diagram images
diagram_imglist = os.listdir(path2_diagram_processed)
print (diagram_imglist)
diagram_img1 = array(Image.open(path2_diagram_processed + '/'+ diagram_imglist[0])) # open one diagram image to get size
diagram_m,diagram_n = diagram_img1.shape[0:2] # get the size of the diagram images
diagram_image_number = len(diagram_imglist) # get the number of diagram images

# create matrix to store all flattened diagram images
diagram_imgmatrix = np.array([np.array(Image.open(path2_diagram_processed + '/' + diagram_img2)).flatten() for diagram_img2 in diagram_imglist])

#Processing for histogram images
histogram_imglist = os.listdir(path2_histogram_processed)
print (histogram_imglist)
histogram_img1 = array(Image.open(path2_histogram_processed + '/'+ histogram_imglist[0])) # open one histogram image to get size
histogram_m,histogram_n = histogram_img1.shape[0:2] # get the size of the histogram images
histogram_image_number = len(histogram_imglist) # get the number of histogram images

# create matrix to store all flattened histogram images
histogram_imgmatrix = np.array([np.array(Image.open(path2_histogram_processed + '/' + histogram_img2)).flatten() for histogram_img2 in histogram_imglist])

#Processing for graph images
graph_imglist = os.listdir(path2_graph_processed)
print (graph_imglist)
graph_img1 = array(Image.open(path2_graph_processed + '/'+ graph_imglist[0])) # open one graph image to get size
graph_m,histogram_n = graph_img1.shape[0:2] # get the size of the graph images
graph_image_number = len(graph_imglist) # get the number of graph images

# create matrix to store all flattened graph images
graph_imgmatrix = array([array(Image.open(path2_graph_processed + '/' + graph_img2)).flatten() for graph_img2 in graph_imglist])

# Combining each matrix
imgmatrix = np.concatenate((plot_imgmatrix, table_imgmatrix), axis=0)
imgmatrix = np.concatenate((imgmatrix, histogram_imgmatrix), axis=0)
imgmatrix = np.concatenate((imgmatrix,graph_imgmatrix), axis=0)
imgmatrix = np.concatenate((imgmatrix,diagram_imgmatrix), axis=0)

class_array = ['Plot','Table','Histogram', 'Graph', 'Diagram']

label=np.ones((num_samples,),dtype = int)
label[0:393]=0
label[393:667]=1
label[667:804]=2
label[804:921]=3
label[921:1029]=4

data,Label = shuffle(imgmatrix,label, random_state=2)
train_data = [data,Label]

(X, y) = (train_data[0],train_data[1])

#split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
X_train = np.stack([X_train]*3, axis=-1)
X_test = np.stack([X_test]*3, axis=-1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols , 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols , 3)


# Data preprocessing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
mean1 = np.mean(X_train) # for finding the mean for centering  to zero
X_train -= mean1
X_test -= mean1


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


model = ResNet50(include_top =False, weights='imagenet', input_shape=(img_cols,img_rows,3), pooling='avg')

x = model.output

# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.7)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu',name='fc-3')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu',name='fc-4')(x)
x = Dropout(0.3)(x)


# a softmax layer for 5 classes
out = Dense(nb_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
resnet_model2 = Model(inputs=model.input, outputs=out)

resnet_model2.summary()

for layer in resnet_model2.layers[:-6]:
	layer.trainable = False


resnet_model2.layers[-1].trainable
 
learning_rate = 0.001
b1 = 0.9
b2 = 0.999

resnet_model2.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=learning_rate, beta_1=b1, beta_2=b2, epsilon=None),metrics=['accuracy'])
y_train = y_train.reshape((-1, 1))
# Fit the model
resnet_model2.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=16, epochs=epoch, verbose=2)
# Final evaluation of the model
scores = resnet_model2.evaluate(X_test, y_test, verbose=1)
print ("%s: %.2f%%" % (resnet_model2.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = resnet_model2.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
resnet_model2.save_weights("model.h5")
print("Saved model to disk")
