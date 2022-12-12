#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:22:26 2022

@author: Mitchell
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import random
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from utils import * 
from importdata import process_gee_point_file


plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.size'] = 13
plt.rcParams["legend.frameon"] = False


random.seed(123)

def normalized_difference(b1,b2):
    return (b1 - b2) / (b1 + b2)
# normalize data over each band and each sampled pt
def z_score(sample):
    mean = np.mean(sample)
    std = np.std(sample)
    z = (sample - mean) / std 
    return z 


#First model training
file1 = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/pointsamples/Point100Sample.csv'
random_sample_df, coordinate_df = process_gee_point_file(file1)
random_sample_df['NDVI'] = normalized_difference(random_sample_df['B8'], random_sample_df['B4']  )

predictors = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6',
            'B7', 'B8', 'B8A', 'B9', 'VH', 'VV', 'angle','landcover', 'elevation', 'NDVI']
# predictors = [ 'B2','B3','B4','B12','NDVI', 'VH', 'VV']
predictand = ['ssm']





# transformed_df  = random_sample_df.groupby(level = 'PointIndex').apply(z_score)
transformed_df  = random_sample_df.apply(z_score)



# define percentages for train, val, test
training_perc = .70

testing_perc = .30

possible_pts = list(coordinate_df.keys())
shuffled = random.sample(possible_pts, k=len(possible_pts))

training_pts = shuffled[0:round(training_perc*len(possible_pts))]
# validation_pts = shuffled[round(training_perc*len(possible_pts)) : round((training_perc+validation_perc)*len(possible_pts)) ]
testing_pts = shuffled[ round(training_perc*len(possible_pts)) : ]

training_df = transformed_df.loc[training_pts]
# validation_df = transformed_df.loc[validation_pts]
testing_df = transformed_df.loc[testing_pts]

X_train = training_df[predictors].to_numpy()
y_train = training_df[predictand].to_numpy()
X_length = y_length = [len(training_df.loc[pt]) for pt in training_pts]

# X_val = validation_df[predictors].to_numpy()
# y_val = validation_df[predictand].to_numpy()

X_test = testing_df[predictors].to_numpy()
y_test = testing_df[predictand].to_numpy()


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


# set hyperparameters
n_neuron       = 64
activation     = 'relu'
num_epochs     = 50
learning_rate  = 0.001
minibatch_size = 64
model_num      = 1
slider         = 5

lstm_model = Sequential()
lstm_model.add(LSTM(n_neuron,input_shape=(X_train.shape[1],X_train.shape[2]),
               return_sequences=True,activation=activation))
lstm_model.add(LSTM(n_neuron,return_sequences=False,
               activation=activation))
lstm_model.add(Dense(n_neuron,activation=activation))
lstm_model.add(Dense(y_train.shape[-1],activation='linear')) 

lstm_model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

lstm_model.summary()


# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = lstm_model.fit(X_train, y_train, 
                        batch_size = minibatch_size,
                        epochs = num_epochs,
                        validation_split=0.2, verbose=1,
                        callbacks=[early_stop],
                        shuffle=False)


y_test_pre = lstm_model.predict(X_test)
y_truth = y_test[0:-slider+1]

rmse = metrics.mean_squared_error(y_truth, y_test_pre, squared = False)


plt.scatter(y_truth,y_test_pre, s = 20)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('RMSE = {}'.format(rmse))
plt.show()


# y_test_pre = xr.Dataset(coords={'time': X_test_xr.time.values[slider-1:], 
#                                'latitude': X_test_xr.latitude.values, 
#                                'longitude': X_test_xr.longitude.values},
#                        data_vars=dict(tas=(['time', 'latitude', 'longitude'], y_test_pre)))


# fig, axes = plt.subplots(figsize=(15,12),ncols=2,nrows=3)



