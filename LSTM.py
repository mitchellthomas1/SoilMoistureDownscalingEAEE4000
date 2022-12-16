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
num_epochs     = 50
learning_rate  = 0.001
minibatch_size = 64
model_num      = 1
slider         = 5
dropout = 0.25
    
X_train, y_train, X_test, y_test, X_length = prepare_data(file1, predictors, predictand)

# prepare shape of lstm
start = np.cumsum(X_length) - X_length
end   = np.cumsum(X_length)

slider = 5
X_train_all = []
y_train_all = []

for i in range(len(X_length)):
    
    X_subset = X_train[start[i]:end[i],:]
    y_subset = y_train[start[i]:end[i],:]
    
    X_subset = np.array([X_subset[i:i+slider] for i in range(0, X_length[i]-slider+1)])
    y_subset = np.array([[y_subset[i+slider-1]] for i in range(0, X_length[i]-slider+1)])
    
    X_train_all.append(X_subset)
    y_train_all.append(y_subset)
    
X_train = np.concatenate(X_train_all,axis=0)
y_train = np.concatenate(y_train_all,axis=0)
X_test  = np.array([X_test[i:i+slider] for i in range(0, X_test.shape[0]-slider+1)])

print('X_train shape, y_train shape, X_test shape\n', X_train.shape,y_train.shape,X_test.shape)



lstm_model = Sequential()
lstm_model.add(LSTM(n_neuron,input_shape=(X_train.shape[1],X_train.shape[2]),
               return_sequences=True,activation=activation, dropout=dropout))
lstm_model.add(LSTM(n_neuron,return_sequences=False,
               activation=activation, dropout=dropout))

lstm_model.add(Dense(n_neuron,activation=activation))
lstm_model.add(Dense(y_train.shape[-1],activation='linear')) 

lstm_model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

lstm_model.summary()


# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = lstm_model.fit(X_train, y_train, 
                        batch_size = minibatch_size,
                        epochs = num_epochs,
                        validation_split=0.2, verbose=1,
                        callbacks=[early_stop],
                        shuffle=False)

y_train_pre = lstm_model.predict(X_train)
y_train_truth = y_train[:,:,0]


y_test_pre = lstm_model.predict(X_test)
y_truth = y_test[0:-slider+1]

rmse = metrics.mean_squared_error(y_train_truth, y_train_pre, squared = False)


plt.scatter(y_train_truth,y_train_pre, s = 1)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('Training RMSE = {}'.format(rmse))
plt.show()




y_test_pre = lstm_model.predict(X_test)
y_truth = y_test[0:-slider+1]

rmse = metrics.mean_squared_error(y_truth, y_test_pre, squared = False)


plt.scatter(y_truth,y_test_pre, s = 1)
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
