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


from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import RandomizedSearchCV
from utils import * 
from buildmodel import prepare_data


file1 = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/pointsamples/Point100Sample.csv'

predictors = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6',
            'B7', 'B8', 'B8A', 'B9', 'VH', 'VV', 'angle','landcover', 'elevation', 'NDVI']
# predictors = [ 'B2','B3','B4','B12','NDVI', 'VH', 'VV']
predictand = ['ssm']


X_train, y_train, X_test, y_truth, X_length = prepare_data(file1, predictors, predictand)

y_train = y_train.flatten()
y_truth = y_truth.flatten()


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 5)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5,55, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [5, 10, 15, 25]

# Minimum number of samples required at each leaf node
min_samples_leaf = [4, 8, 12,16]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

reg0 = RandomForestRegressor(random_state=0)
# perform cross validation
rf_random0 = RandomizedSearchCV(estimator = reg0, param_distributions = random_grid, 
                                n_iter = 5, cv = 3, verbose=2, n_jobs = -1)
rf_tas = rf_random0.fit(X_train,y_train)

print("The best hyperparameters: \n",rf_tas.best_params_)


y_test_pre = rf_tas.predict(X_test)

rmse = metrics.mean_squared_error(y_truth, y_test_pre, squared = False)


plt.scatter(y_truth,y_test_pre, s = 20)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('RMSE = {}'.format(rmse))
plt.show()
plot_history(rf_random0)

# y_test_pre = xr.Dataset(coords={'time': X_test_xr.time.values[slider-1:], 
#                                'latitude': X_test_xr.latitude.values, 
#                                'longitude': X_test_xr.longitude.values},
#                        data_vars=dict(tas=(['time', 'latitude', 'longitude'], y_test_pre)))


# fig, axes = plt.subplots(figsize=(15,12),ncols=2,nrows=3)
