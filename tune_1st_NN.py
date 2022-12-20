#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:49:28 2022

@author: Mitchell
"""
from tensorflow import keras
import tensorflow as tf
import sys
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


# from utils import * 



def data():
    from buildmodel import prepare_10kmdata, get_landcover_bands
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
    #blah blah blah
    from tensorflow.keras.models import Model
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
    if layers_choice == 'three':
        x = dense3(x)
    x = dropout_layer(x)
    output1 = output_layer_1(x)
    model = Model(inputs = input1, outputs = output1)
    model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_choice))
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    model.fit(X_train, Y_train, 
                        batch_size      = minibatch_size,
                        epochs          = epoch_choice,
                        validation_data=(X_test, Y_test),
                        verbose         = 2,
                        callbacks       = [early_stop])
    loss = model.evaluate(X_test,  Y_test, verbose=0)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}
    

    


if __name__ == '__main__':
    print('RUNNING WITH {} MAX_EVALS\n\n\n'.format(sys.argv[1]))
    best_run, best_model = optim.minimize(model=model,
                                      data= data,
                                      algo= tpe.suggest,
                                      max_evals=int(sys.argv[1]),
                                      trials= Trials(),
                                      notebook_name = 'FusedModel1',
                                      eval_space = True)
    print('\n\n\nBest Run Parameters: ')
    print(best_run)



