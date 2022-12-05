#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:53:17 2022

@author: Mitchell
"""

import ee
import pandas as pd
import geemap as gm
import matplotlib.pyplot as plt
ee.Initialize()

start = '2017-01-01'
end = '2022-01-01'

nan_value = -99999

# input data sources:
   
S1_raw = ee.ImageCollection("COPERNICUS/S1_GRD")
S2_raw = ee.ImageCollection("COPERNICUS/S2")
SMAP = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")

# input projection
crs = 'EPSG:4326'
transform = [0.140625,0,-179.9993125,0,-0.09375,90.000125]

s1_crs = 'EPSG:32615'
s1_scale = 10

# Geomeetries of study
geometries = ee.FeatureCollection('users/mlt2177/SoilMoistureDownscale/10testgeoms4')

date_index = pd.date_range(start=start, end=end, freq = '10D')

# for feature in geometries.getInfo()['features']:
#     geom = ee.Feature(feature).geometry()
#     for i in range(len(date_index) - 1):
    
feature = geometries.getInfo()['features'][0]
geom = ee.Feature(feature).geometry().buffer(-1)
i = 0
start,end = str(date_index.values[i]), str(date_index.values[i+1])
s1_mean = S1_raw.filterDate(start,end).filterBounds(geom).mean()
s2_mean = S1_raw.filterDate(start,end).filterBounds(geom).mean()
smap_mean = SMAP.filterDate(start,end).filterBounds(geom).mean()
# Reduce Resolution to smap
s1_reduce = s1_mean.setDefaultProjection(crs = s1_crs, scale = s1_scale) \
                    .reduceResolution(reducer = ee.Reducer.mean(),
                         bestEffort = True, 
                         maxPixels = 65000) \
                    .reproject(crs = crs, crsTransform = transform)
s1_np = gm.ee_to_numpy(s1_reduce, bands=['VV','VH'], region=geom, properties=None, default_value=nan_value)
        # print(s1_np)
        # s2_reduce = s2_mean.reduceResolution(reducer = ee.Reducer.mean(),
        #                           bestEffort = True, 
        #                           maxPixels = 65000)
        
        
    
    
    

# ee_dates = ee.List(date_index.values)