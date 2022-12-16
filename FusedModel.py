#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:39:16 2022

@author: Mitchell
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential, Input
from tensorflow.keras.utils import plot_model
from utils import * 
from buildmodel import prepare_10kmdata, prepare_40mdata, get_landcover_bands

path = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/'

file1 =  path + 'DataDownload/pointsamples/Point100Sample.csv'
# data file
file2 = path + 'DataDownload/InSitu/InSituGEEOutputs/SMGaugePoints2.csv'

# STEP 1 Build first neural net
lc_bands = list(get_landcover_bands([file1,file2]))
predictors = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'VH', 'VV', 'angle', 'elevation', 'NDVI']
# predictors = [ 'B2','B3','B4','B12','NDVI', 'VH', 'VV']
predictand = ['ssm']

# set hyperparameters
n_neuron       = 64
activation     = 'relu'
num_epochs     = 10
learning_rate  = 0.001
minibatch_size = 64
model_num      = 1
    
X_train, y_train, X_test, y_test, X_length = prepare_10kmdata(file1, predictors, predictand, lc_bands, dropna = True)



input1 = Input(shape= (X_train.shape[1],))
# Layer 1
dense1 = Dense(n_neuron,  activation=activation)
x = dense1(input1)
# Layer 2
dense2 = Dense(n_neuron,  activation=activation)
x = dense2(x)
#Layer 3
dense3 = Dense(n_neuron,  activation=activation)
x = dense3(x)
#Output
output1 = Dense(y_train.shape[1],  activation='linear')(x)


model1 = Model(inputs = input1, outputs = output1)
model1.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
model1.summary()
# j = plot_model(model1, path + "my_first_model.png")

# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


history = model1.fit(X_train, y_train, 
                    batch_size      = minibatch_size,
                    epochs          = num_epochs,
                    validation_split= 0.2, 
                    verbose         = 1,
                    callbacks       = [early_stop])



# Step 2, build 2nd neural net


input2 = Input(shape= (1,))

input_concat = Concatenate()([output1, input2 ])

dense1_2 = Dense(n_neuron,  activation=activation)
x2 = dense1_2(input_concat)
# Layer 2
dense2_2 = Dense(n_neuron,  activation=activation)(x2)
#Layer 3
dense3_2 = Dense(n_neuron,  activation=activation)(dense2_2)

output2 = Dense(y_train.shape[1],  activation='linear')(dense3_2)

model2 = Model(inputs = [input1, input2], outputs = output2)

#set other layers as trainable
dense1.Trainable = False
dense2.trainable = False
dense3.trainable = True


model2.compile(loss='mse', optimizer='sgd')

model2.summary()

plot_model(model2)


# model2.

# STEP 2: bring in additional data at 30m resolution
insitu_file = path+'DataDownload/InSitu/SoilMoistureDataFrameGreaterThan80.csv'
predictors = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'VH', 'VV', 'angle', 'elevation', 'NDVI','ssm']
predictand = ['InSituSM']
X2_train, y2_train, X2_test, y2_test, X2_length = prepare_40mdata(file2,insitu_file, predictors, predictand, lc_bands, dropna = True)

# model.predict()


X_train2_i1 = np.random.randn(*X_train.shape)
X_train2_i2 = np.random.randn(X_train.shape[0], 1)
X_train2_o2 = np.random.randn(X_train.shape[0], 1)

history2 = model2.fit([X_train2_i1, X_train2_i2], y_train, 
                    batch_size      = minibatch_size,
                    epochs          = num_epochs,
                    validation_split= 0.2, 
                    verbose         = 1,
                    callbacks       = [early_stop])




# y_train_pre = model1.predict(X_train)
# y_train_truth = y_train

# rmse = metrics.mean_squared_error(y_train_truth, y_train_pre, squared = False)



# plt.scatter(y_train_truth,y_train_pre, s = 2)
# plt.xlabel('Truth')
# plt.ylabel('Predicted')
# plt.xlim(-3,3)
# plt.ylim(-3,3)
# plt.title('Training RMSE = {}'.format(rmse))
# plt.show()




# y_test_pre = model1.predict(X_test)
# y_truth = y_test

# rmse = metrics.mean_squared_error(y_truth, y_test_pre, squared = False)


# plt.scatter(y_truth,y_test_pre, s = 2)
# plt.xlabel('Truth')
# plt.ylabel('Predicted')
# plt.xlim(-3,3)
# plt.ylim(-3,3)
# plt.title('Testing RMSE = {}'.format(rmse))
# plt.show()
# plt.clf()


# plot_history(history)
# plt.show()
# plt.clf()

