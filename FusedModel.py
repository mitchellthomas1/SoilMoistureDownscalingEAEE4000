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
from buildmodel import prepare_10kmdata, prepare_40mdata, get_landcover_bands,z_score

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
num_epochs     = 15
learning_rate  = 0.0005
minibatch_size = 64
model_num      = 1


# Bring in data for part I
X_train, y_train, X_test, y_test, X_length = prepare_10kmdata(file1, predictors, predictand, lc_bands, dropna = True)

# Bring in data for part II (30m resolution)
insitu_file = path+'DataDownload/InSitu/SoilMoistureDataFrameGreaterThan80.csv'
predictors1 = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'VH', 'VV', 'angle', 'elevation', 'NDVI']
predictors2 = ['ssm']
predictand = ['InSituSM']

X2_1_train, X2_2_train, y2_train, X2_1_test, X2_2_test, y2_test, X2_length = prepare_40mdata(
                            file2,insitu_file, predictors1, predictors2, predictand, lc_bands, dropna = True)


input1 = Input(shape= (X_train.shape[1],))
# Layer 1
dense1 = Dense(n_neuron,  activation=activation)

# Layer 2
dense2 = Dense(n_neuron,  activation=activation)

#Layer 3
dense3 = Dense(n_neuron,  activation=activation)

#Dropout
# dropout_layer = Dropout(rate = 0.3)

#Output
output_layer_1 = Dense(y_train.shape[1],  activation='linear')

x = dense1(input1)
x = dense2(x)
x = dense3(x)
# x = dropout_layer(x)
output1 = output_layer_1(x)

model1 = Model(inputs = input1, outputs = output1)
model1.compile(loss='mse', metrics=['accuracy'],
               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
model1.summary()
# j = plot_model(model1, path + "my_first_model.png")

# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


history1 = model1.fit(X_train, y_train, 
                    batch_size      = minibatch_size,
                    epochs          = num_epochs,
                    validation_split= 0.2, 
                    verbose         = 2,
                    callbacks       = [early_stop])

# Evaluate the performance of this first testing:
y_test_pre = model1.predict(X_test)
y_truth = y_test
rmse = metrics.mean_squared_error(y_truth, y_test_pre, squared = False)
plt.scatter(y_truth,y_test_pre, s = 2)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('Performance of first half model for predicting SMAP = {}'.format(rmse))
plt.show()
plt.clf()



# See how this first testing does on its own for predicting in situ!
y_test_pre = model1.predict(X2_1_test)
y_truth = y2_test
rmse = metrics.mean_squared_error(y_truth, y_test_pre, squared = False)
plt.scatter(y_truth,y_test_pre, s = 2)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('First Half Model vs In Situ Testing RMSE = {}'.format(rmse))
plt.show()
plt.clf()





# Step 2, build 2nd neural net

input2 = Input(shape= (1,))

input_concat = Concatenate()([output1, input2 ])
dense1_2 = Dense(n_neuron,  activation=activation)
# Layer 2
dense2_2 = Dense(n_neuron,  activation=activation)
# Layer 3
dense3_2 = Dense(n_neuron,  activation=activation)
#output
output_layer_2 = Dense(y_train.shape[1],  activation='linear')
#dropout
# dropout_layer = Dropout(rate = 0.3)


x = dense1_2(input_concat)
x = dense2_2(x)
x = dense3_2(x)
# x = dropout_layer(x)
output2 = output_layer_2(x)

model2 = Model(inputs = [input1, input2], outputs = output2)

#set other layers as trainable
dense1.Trainable = False
dense2.trainable = False
dense3.trainable = True






plot_model(model2)


# model2.



# model.predict()

# set hyperparameters
n_neuron       = 64
activation     = 'relu'
num_epochs     = 50
learning_rate  = 0.0001
minibatch_size = 64


model2.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
model2.summary()

# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history2 = model2.fit([X2_1_train, X2_2_train], y2_train, 
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


y_truth = y2_test

y_smap = X2_2_test


rmse = metrics.mean_squared_error(y_truth, y_smap, squared = False)


plt.scatter(y_truth,y_smap, s = 2)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('SMAP vs In Situ, Testing RMSE = {}'.format(rmse))
plt.show()
plt.clf()





y_test_pre = model2.predict([X2_1_test, X2_2_test])


rmse = metrics.mean_squared_error(y_truth, y_test_pre, squared = False)


plt.scatter(y_truth,y_test_pre, s = 2)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('Full Model vs In Situ Testing RMSE = {}'.format(rmse))
plt.show()
plt.clf()





plot_history(history2)
plt.show()
plt.clf()



# insitu = pd.read_csv(insitu_file)
# insitu.index = pd.to_datetime(insitu['Unnamed: 0'])
# insitu = insitu['InSituSM']
# starts = np.where(insitu.index == pd.Timestamp(2019,1,1))[0]
# testingstarts = starts[round(0.80*len(starts))]
# z_insitu = z_score(insitu)
# y2_test_check = z_insitu[testingstarts:]
# print(y2_test)
# print(y2_test_check)

