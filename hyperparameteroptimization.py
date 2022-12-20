#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:49:28 2022

@author: Mitchell
"""

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
from tensorflow.keras.utils import plot_model
# from utils import * 
from buildmodel import prepare_10kmdata, prepare_40mdata, get_landcover_bands,z_score


def tune_1st_NN(max_evals):
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
        #blah blah blah
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
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
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
                                          max_evals=max_evals,
                                          trials= Trials(),
                                          notebook_name = 'FusedModel1',
                                          eval_space = True)
    return best_run, best_model



def tune_2nd_NN(max_evals):
    
    def data():
        path = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/'
        file1 =  path + 'DataDownload/pointsamples/Point100Sample.csv'
        file2 = path + 'DataDownload/InSitu/InSituGEEOutputs/SMGaugePoints2.csv'
        insitu_file = path+'DataDownload/InSitu/SoilMoistureDataFrameGreaterThan80.csv'
        lc_bands = list(get_landcover_bands([file1,file2]))
        predictors = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
               'B8A', 'B9', 'VH', 'VV', 'angle', 'elevation', 'NDVI']
        predictand = ['ssm']
        X1_tr, X2_tr, y2_tr, _,_,_, _ = prepare_40mdata(
                            file2,insitu_file, predictors1, predictors2, predictand, lc_bands, dropna = True)
        index = int(0.80*len(y2_tr))
        X1_train = X1_tr[:index] 
        X2_train = X2_tr[:index] 
        Y_train = y2_tr[:index]
        X1_test = X1_tr[index:] 
        X2_test = X2_tr[index:] 
        Y_test = y2_tr[index:]
        
        return X1_train,X2_train, Y_train, X1_test,X2_test, Y_test

    def model(X1_train,X2_train, Y_train, X1_test,X2_test, Y_test):
        
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras import Input
        
        # hyperparameters from last tuning:
        # set hyperparameters based on tuning
        n_neuron       = 64
        activation     = 'relu'
        num_epochs     = 15
        learning_rate  = 0.0005
        minibatch_size = 64
        model_num      = 1
        dropout_rate = 0
        n_layers = 'three'
        # Model input
        input1 = Input(shape= (X_train.shape[1],))
        dense1 = Dense(n_neuron,  activation=activation)
        dense2 = Dense(n_neuron,  activation=activation)
        dense3 = Dense(n_neuron,  activation=activation)
        dropout_layer = Dropout(rate = dropout_rate)
        output_layer_1 = Dense(y_train.shape[1],  activation='linear')
        x = dense1(input1)
        x = dense2(x)
        if n_layers == 3:
            x = dense3(x)
        x = dropout_layer(x)
        output1 = output_layer_1(x)

        model1 = Model(inputs = input1, outputs = output1)
        model1.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        model.fit(X_train, Y_train, 
                        batch_size      = minibatch_size,
                        epochs          = epoch_choice,
                        validation_data=(X_test, Y_test),
                        verbose         = 2,
                        callbacks       = [early_stop])
        # --- Define layers with hyperparameter tuning built in -----



        # Hyperparameter Tuning options from
        n_neuron_choice = {{choice([16, 32, 64, 128])}}
        learning_rate_choice = {{choice([0.01,0.005, 0.001,0.0005, 0.0001,0.00001])}}
        dropout_choice = {{uniform(0.,1.)}}
        epoch_choice = {{choice([5,10,15,20,30,40,50])}}
        layers_choice = {{choice(['two', 'three'])}}
        trainable = {{choice(['none','last', 'last2', 'last3'])}}
        activation     = 'relu'
        minibatch_size = 64


        input2 = Input(shape= (1,))

        input_concat = Concatenate()([output1, input2 ])
        dense1_2 = Dense(n_neuron_choice,  activation=activation)
        # Layer 2
        dense2_2 = Dense(n_neuron_choice,  activation=activation)
        # Layer 3
        dense3_2 = Dense(n_neuron_choice,  activation=activation)
        #output
        output_layer_2 = Dense(y_train.shape[1],  activation='linear')
        #dropout
        dropout_layer = Dropout(rate = dropout_choice)
        
        dense1.Trainable = False
        dense2.Trainable = False
        dense3.Trainable = False
        if trainable == 'last':
            dense3.Trainable = True
        elif trainable == 'last2':
            dense2.Trainable = True
            dense3.Trainable = True
        elif trainable == 'last3':
            dense1.Trainable = True
            dense2.Trainable = True
            dense3.Trainable = True
            
            
        

        x = dense1_2(input_concat)
        x = dense2_2(x)
        if layers_choice == 'three':
            x = dense3_2(x)
        x = dropout_layer(x)
        output2 = output_layer_2(x)

        model2 = Model(inputs = [input1, input2], outputs = output2)
        
        model.compile(loss='mse',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_choice))

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        model.fit((X1_train,X2_train), Y_train, 
                            batch_size      = minibatch_size,
                            epochs          = epoch_choice,
                            validation_data= (X_test, Y_test),
                            verbose         = 2,
                            callbacks       = [early_stop])


        loss = model.evaluate((X1_test,X2_test),  Y_test, verbose=0)
        print('loss: ' , loss)

        return {'loss': loss, 'status': STATUS_OK, 'model': model}


    # model(*data())
    best_run, best_model = optim.minimize(model=model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=max_evals,
                                              trials=Trials(),
                                              notebook_name = 'FusedModel1',
                                              eval_space=True)
    return best_run, best_model
