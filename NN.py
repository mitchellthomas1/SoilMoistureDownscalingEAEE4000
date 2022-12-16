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
from tensorflow.keras import Sequential
from utils import * 
from buildmodel import prepare_data


file1 = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/pointsamples/Point100Sample.csv'


predictors = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'VH', 'VV', 'angle', 'elevation', 'NDVI', 'x0_11',
       'x0_21', 'x0_41', 'x0_42', 'x0_43', 'x0_52', 'x0_71', 'x0_81', 'x0_82',
       'x0_90', 'x0_95']
# predictors = [ 'B2','B3','B4','B12','NDVI', 'VH', 'VV']
predictand = ['ssm']

# set hyperparameters
n_neuron       = 64
activation     = 'relu'
num_epochs     = 10
learning_rate  = 0.001
minibatch_size = 64
model_num      = 1
    
X_train, y_train, X_test, y_test, X_length = prepare_data(file1, predictors, predictand)




model = Sequential()

model.add(Dense(n_neuron,  activation=activation,input_shape=(X_train.shape[1],))) #  the 1st hidden layer 
model.add(Dense(n_neuron,  activation=activation)) # the 2nd hidden layer
model.add(Dense(n_neuron,  activation=activation)) # the 3rd hidden layer
model.add(Dense(y_train.shape[1],  activation='linear')) # the output layer


model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

model.summary()


# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


history = model.fit(X_train, y_train, 
                    batch_size      = minibatch_size,
                    epochs          = num_epochs,
                    validation_split= 0.2, 
                    verbose         = 1,
                    callbacks       = [early_stop])

y_train_pre = model.predict(X_train)
y_train_truth = y_train


y_test_pre = model.predict(X_test)
y_truth = y_test

rmse = metrics.mean_squared_error(y_train_truth, y_train_pre, squared = False)



plt.scatter(y_train_truth,y_train_pre, s = 2)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('Training RMSE = {}'.format(rmse))
plt.show()




y_test_pre = model.predict(X_test)
y_truth = y_test

rmse = metrics.mean_squared_error(y_truth, y_test_pre, squared = False)


plt.scatter(y_truth,y_test_pre, s = 2)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('Testing RMSE = {}'.format(rmse))
plt.show()
plt.clf()


plot_history(history)
plt.show()
plt.clf()


# y_test_pre = xr.Dataset(coords={'time': X_test_xr.time.values[slider-1:], 
#                                'latitude': X_test_xr.latitude.values, 
#                                'longitude': X_test_xr.longitude.values},
#                        data_vars=dict(tas=(['time', 'latitude', 'longitude'], y_test_pre)))


# fig, axes = plt.subplots(figsize=(15,12),ncols=2,nrows=3)
