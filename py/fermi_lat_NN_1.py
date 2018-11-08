

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Reshape
import matplotlib.pyplot as plt
from os import listdir
from keras import backend as k
import sys
from random import shuffle
from astropy.io import fits
from numpy import vstack
from keras.utils import to_categorical

file_1=fits.open('input3')
file_2=fits.open('OUTPUT')

#load dataset
input_1  = np.empty((1905,5))
input_1=vstack(file_1[1].data)
print(input_1.shape)
output_1 = np.empty((1905,1))

#file_1=np.array(file_1)
#print(file_1.shape)
#input1=np.array(file_1[1].data)
output_1=vstack(file_2[1].data)
#print(input1.shape)


#shuffle(targets)
print(input_1.shape)
model=keras.Sequential()
for i in range(1905):
    if output_1[i,:]=="AGN":
        output_1[i,:]=1
    else: output_1[i,:]=0.5
#output_1=to_categorical(output_1)

input_1=np.append(input_1,output_1,axis=1)
np.random.shuffle(input_1)
test=input_1[:1600,0:5]
test_truth=input_1[:1600,5:]
truth_inp=input_1[1600:,:5]
truth_out=input_1[1600:,5:]
print(test.shape)

#print(input_1)

#model.add(keras.layers.Reshape((1000*5,),input_shape=(5,)))
#model.add(keras.layers.Conv2D(filters=4,kernel_size=5,strides=(2,2),padding='same',activation=keras.layers.LeakyReLU(alpha=0.2),input_shape=(64,64,3,),data_format="channels_last"))
#model.add(keras.layers.Conv2D(filters=8,kernel_size=5,strides=(2,2),padding='same',activation=keras.layers.LeakyReLU(alpha=0.2),data_format="channels_last"))
#model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Conv2D(filters=16,kernel_size=5,strides=(2,2),padding='same',activation=keras.layers.LeakyReLU(alpha=0.2),data_format="channels_last"))
#model.add(keras.layers.Conv2D(filters=32,kernel_size=3,strides=(2,2),padding='same',activation=keras.layers.LeakyReLU(alpha=0.2),data_format="channels_last"))
#model.add(keras.layers.Conv2D(filters=64,kernel_size=3,strides=(2,2),padding='same',activation=keras.layers.LeakyReLU(alpha=0.2),data_format="channels_last"))
#model.add(keras.layers.Conv2D(filters=3,kernel_size=10,strides=(2,2),padding='same',activation='relu',data_format="channels_last"))

#model.add(keras.layers.UpSampling1D(size=2,input_shape=(5,)))
#model.add(keras.layers.Conv1D(filters=2,input_shape=(5,),kernel_size=3,strides=1,padding='same',activation='tanh'))
#model.add(keras.layers.Dropout(rate=0.1,input_shape=(5,)))
model.add(keras.layers.Dense(5,input_shape=(5,),activation='tanh'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(3,activation='tanh'))
model.add(keras.layers.Dense(1,activation='tanh'))
#model.add(keras.layers.Dense(1,activation='tanh'))
#model.add(keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"))
#model.add(keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=1,padding='same',activation = keras.layers.LeakyReLU(alpha=0),data_format="channels_last"))
#model.add(keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"))
#model.add(keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=1,padding='same',activation=keras.layers.LeakyReLU(alpha=0),data_format="channels_last"))
#model.add(keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"))
#model.add(keras.layers.Conv2DTranspose(filters=16,kernel_size=5,strides=1,padding='same',activation=keras.layers.LeakyReLU(alpha=0),data_format="channels_last"))
#model.add(keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"))
#model.add(keras.layers.Conv2DTranspose(filters=8,kernel_size=5,strides=1,padding='same',activation=keras.layers.LeakyReLU(alpha=0),data_format="channels_last"))
#model.add(keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"))
#model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Conv2DTranspose(filters=4,kernel_size=5,strides=1,padding='same',activation=keras.layers.LeakyReLU(alpha=0),data_format="channels_last"))
#model.add(keras.layers.Reshape((64,64,3)))
#model.add(keras.layers.Conv2DTranspose(filters=3,kernel_size=5,strides=1,padding='same',activation=keras.layers.LeakyReLU(alpha=0),data_format="channels_last"))
#model.add(keras.layers.Dense(3*64*64))
#model.add(keras.layers.Reshape((64,64,3)))
#model.add(keras.layers.Conv2D(filters=3,kernel_size=5,padding="same",data_format="channels_first"))
#configure the model
model.compile(optimizer=tf.train.AdamOptimizer(0.0001),loss='binary_crossentropy', metrics=['accuracy','mae'])
mod=model.fit(test,test_truth,epochs=25,batch_size=200,validation_split=0.3)
predictions = model.predict(truth_inp, batch_size=1)
#print(predictions.shape)
predictions=vstack(predictions)
#print(predictions)
#print(truth_out)
#predictions_err=predictions-truth_out
#print(predictions_err)
#np.savetxt('try.txt',predictions)
#np.savetxt('try1.txt',truth_out)
predictions=np.ravel(predictions)
truth_out=np.ravel(truth_out)
print(predictions.shape)
print(truth_out.shape)
print(predictions)
plt.plot(predictions,truth_out,'o')
plt.show()
#plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

#his=mod.history["loss"]
#plt.plot(his)
model.summary()
