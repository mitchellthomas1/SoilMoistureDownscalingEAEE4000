import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import pearsonr
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
# from utils import * 
from buildmodel import prepare_10kmdata, prepare_40mdata, get_landcover_bands,z_score

#define file path and files
#file1: satellite data for randomly sampled points across U.S.
#file2: satellite export data associated with in situ points

path = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/'
file1 =  path + 'DataDownload/pointsamples/Point100Sample.csv'
# data file
file2 = path + 'DataDownload/InSitu/InSituGEEOutputs/SMGaugePoints2.csv'

 # get the landcover bands that will be used in this study. This is needed for future one hot encoding.
lc_bands = list(get_landcover_bands([file1,file2]))
# define predictors and predictand in file
predictors = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'VH', 'VV', 'angle', 'elevation', 'NDVI']
predictand = ['ssm']


# Bring in data for part I (10km resolution)
X_train, y_train, X_test, y_test, X_length = prepare_10kmdata(file1, predictors, predictand, lc_bands, dropna = True)

# Bring in data for part II (30m resolution) (in order to conduct preliminary evaluation of first neural net)
insitu_file = path+'DataDownload/InSitu/SoilMoistureDataFrameGreaterThan80.csv'
predictors1 = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'VH', 'VV', 'angle', 'elevation', 'NDVI']
predictors2 = ['ssm']
predictand = ['InSituSM']
X2_1_train, X2_2_train, y2_train, X2_1_test, X2_2_test, y2_test, X2_length = prepare_40mdata(
                            file2,insitu_file, predictors1, predictors2, predictand, lc_bands, dropna = True)
def data():
    path = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/'
    file1 =  path + 'DataDownload/pointsamples/Point100Sample.csv'
    file2 = path + 'DataDownload/InSitu/InSituGEEOutputs/SMGaugePoints2.csv'
    lc_bands = list(get_landcover_bands([file1,file2]))
    predictors = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
           'B8A', 'B9', 'VH', 'VV', 'angle', 'elevation', 'NDVI']
    predictand = ['ssm']
    X_t, y_t, _, _, _ = prepare_10kmdata(file1,
                     predictors, predictand, lc_bands, dropna = True)
    index = int(0.80*len(X_t))
    X_train = X_t[:index] 
    Y_train = y_t[:index]
    X_test = X_t[index : ]
    Y_test = y_t[index :]
    return X_train, Y_train,X_test, Y_test 

def model(X_train, Y_train,X_test, Y_test):
    # --- Define layers with hyperparameter tuning built in -----

    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import Input

    # Hyperparameter Tuning options
    n_neuron_choice = {{choice([16, 32, 64, 128])}}
    learning_rate_choice = {{choice([0.01,0.005, 0.001,0.0005, 0.0001,0.00001])}}
    dropout_choice = {{uniform(0.,1.)}}
    epoch_choice = {{choice([5,10,15,20,30,40,50])}}
    layers_choice = {{choice(['two', 'three'])}}
    activation     = 'relu'
    minibatch_size = 64
    # Model input
    input1 = Input(shape= (X_train.shape[1],))
    # Layer 1
    dense1 = Dense(n_neuron_choice,  activation=activation)
    # Layer 2
    dense2 = Dense(n_neuron_choice,  activation=activation)
    #Layer 3
    dense3 = Dense(n_neuron_choice,  activation=activation)
    # Dropout 
    dropout_layer = Dropout(dropout_choice)
    #Output
    output_layer_1 = Dense(Y_train.shape[1],  activation='linear')
    # --- Build Model ---
    x = dense1(input1)
    x = dense2(x)
    x = dense3(x)
    x = dropout_layer(x)
    output1 = output_layer_1(x)
    model = Model(inputs = input1, outputs = output1)
    model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_choice))
    model.fit(X_train, Y_train, 
                        batch_size      = minibatch_size,
                        epochs          = epoch_choice,
                        validation_data=(X_test, Y_test),
                        verbose         = 2,
                        callbacks       = [early_stop])
    loss = model.evaluate(X_test,  Y_test, verbose=0)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=model,
                                          data= data,
                                          algo= tpe.suggest,
                                          max_evals=100,
                                          trials= Trials(),
                                          notebook_name = 'FusedModel1',
                                          eval_space = True,
                                          verbose =True)
