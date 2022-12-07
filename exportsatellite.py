#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:53:17 2022

@author: Mitchell
"""
import os
import ee
import pandas as pd
import geemap as gm
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from netCDF4 import date2num,num2date
ee.Initialize()



# create netcdf file
def create_netcdf(region_i):
    try: 
        ds.close()  # just to be safe, make sure dataset is not already open.
    except: 
        pass
    fn = '/Volumes/SeagateExternalDrive/MLEnvironment/Exports/export1/Region{}/s2s1smapexport1.nc'.format(region_i)
    print(fn )
    
    ds = nc.Dataset(fn, 'w', format='NETCDF4', clobber=False)
    # add dimensions
    time = ds.createDimension('time', None)
    lat = ds.createDimension('lat', 7)
    lon = ds.createDimension('lon', 7)
    
    times = ds.createVariable('time', 'f4', ('time',))
    times.units = 'hours since 1800-01-01'
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    B1 = ds.createVariable('B1', np.float64, ('time', 'lat', 'lon',))
    B2 = ds.createVariable('B2', np.float64, ('time', 'lat', 'lon',))
    B3 = ds.createVariable('B3', np.float64, ('time', 'lat', 'lon',))
    B4 = ds.createVariable('B4', np.float64, ('time', 'lat', 'lon',))
    B5 = ds.createVariable('B5', np.float64, ('time', 'lat', 'lon',))
    B6 = ds.createVariable('B6', np.float64, ('time', 'lat', 'lon',))
    B7 = ds.createVariable('B7', np.float64, ('time', 'lat', 'lon',))
    B8 = ds.createVariable('B8', np.float64, ('time', 'lat', 'lon',))
    B8A = ds.createVariable('B8A', np.float64, ('time', 'lat', 'lon',))
    B9 = ds.createVariable('B9', np.float64, ('time', 'lat', 'lon',))
    B10 = ds.createVariable('B10', np.float64, ('time', 'lat', 'lon',))
    B11 = ds.createVariable('B11', np.float64, ('time', 'lat', 'lon',))
    B12 = ds.createVariable('B12', np.float64, ('time', 'lat', 'lon',))
    VV = ds.createVariable('VV', np.float64, ('time', 'lat', 'lon',))
    VH = ds.createVariable('VH', np.float64, ('time', 'lat', 'lon',))
    Angle = ds.createVariable('angle', np.float64, ('time', 'lat', 'lon',))
    SSM = ds.createVariable('ssm', np.float64, ('time', 'lat', 'lon',))
    
    tlonlat = (times, lons, lats)
    attributes = (B1, B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12,VV,VH,Angle, SSM)

    return ds, tlonlat, attributes




start = '2019-01-01'
end = '2022-01-01'

nan_value = -99999

# input data sources:
   
S1_raw = ee.ImageCollection("COPERNICUS/S1_GRD")
S2_raw = ee.ImageCollection("COPERNICUS/S2")
SMAP = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")


# input projection
crs = 'EPSG:4326'
transform = [0.140625,0,-179.9993125,0,-0.09375,90.000125]

smap_crs = 'EPSG:4326'
smap_transform = [0.140625,0,-179.9993125,0,-0.09375,90.000125]


s1_crs = 'EPSG:32615'
s1_scale = 10

s2_crs = 'EPSG:32615'
s2_scale = 20

# Geomeetries of study
geometries = ee.FeatureCollection('users/mlt2177/SoilMoistureDownscale/10testgeoms4')

# bands to study
s1_bands = ['VV', 'VH', 'angle']
s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

# cloud masking for s2
s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
max_cloud_prob = 65
def mask_clouds(s2_image):
    cloud_image = s2_clouds.filter( ee.Filter.eq('system:index', s2_image.get('system:index'))).first()
    mask = cloud_image.lt(max_cloud_prob)
    return s2_image.updateMask(mask)

s2_btypes = S2_raw.select(s2_bands).filterDate(start,end).filterBounds(geometries) \
        .first().bandTypes().getInfo()
for b in s2_btypes.keys():
    s2_btypes[b] = ee.PixelType('double', -99999,65535)
def s2CastTypes(s2_image):
    s2_cast = s2_image.cast(s2_btypes)
    return s2_cast
    

def ndvi(image):
    ndvi = image.normalizedDifference('B')

date_index = pd.date_range(start=start, end=end, freq = '10D')
s2_filtered = S2_raw.filterDate(start,end).filterBounds(geometries) \
                            .select(s2_bands).map(mask_clouds).map(s2CastTypes)

for j, feature in enumerate(geometries.getInfo()['features']):
    ds,tll, attributes =  create_netcdf(j+1)
    times, lons, lats = tll
    
    times[:] = date2num(date_index.values, times.units)
    
    geom = ee.Feature(feature).geometry().buffer(-1)
    # get lat and lon
    lon_lat = ee.Image.pixelLonLat().setDefaultProjection(crs = smap_crs, crsTransform = smap_transform)
    lon_lat_np = gm.ee_to_numpy(lon_lat,bands = ['longitude', 'latitude'], region=geom, properties=None, default_value=nan_value)
    lons[:] = lon_lat_np[0,:,0]
    lats[:] = lon_lat_np[:,0,1]

    
    for i in range(len(date_index) - 1):
        start,end = str(date_index.values[i]), str(date_index.values[i+1])
        s1_mean = S1_raw.filterDate(start,end).filterBounds(geom).mean()
        s2_mean = s2_filtered.filterDate(start,end).filterBounds(geom).mean()
        
                        #.map(mask_clouds).map(s2CastTypes).mean()
        smap_mean = SMAP.filterDate(start,end).filterBounds(geom).mean()
        # Reduce Resolution to smap
        s1_reduce = s1_mean.setDefaultProjection(crs = s1_crs, scale = s1_scale) \
                            .reduceResolution(reducer = ee.Reducer.mean(),
                                 bestEffort = True, 
                                 maxPixels = 65000) \
                            .reproject(crs = crs, crsTransform = transform)
                            
        s1_np = gm.ee_to_numpy(s1_reduce, bands=s1_bands, region=geom, properties=None, default_value=nan_value)
        
        s2_reduce = s2_mean.setDefaultProjection(crs = s2_crs, scale = s2_scale)\
                        .reduceResolution(reducer = ee.Reducer.mean(),
                                 bestEffort = True, 
                                 maxPixels = 65000) \
                            .reproject(crs = crs, crsTransform = transform)
        s2_np_1_7 = gm.ee_to_numpy(s2_reduce,bands = s2_bands[:7], region=geom, properties=None, default_value=nan_value)
        s2_np_8_13 = gm.ee_to_numpy(s2_reduce,bands = s2_bands[7:], region=geom, properties=None, default_value=nan_value)
        smap_np = gm.ee_to_numpy(smap_mean,bands = ['ssm'], region=geom, properties=None, default_value=nan_value)
        
        # (B1, B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12,VV,VH,Angle, SSM)
        data_concat_arr = np.concatenate([s2_np_1_7 , s2_np_8_13, s1_np, smap_np],axis = 2)
        
        for k, att in enumerate(attributes):
            att[i,:,:] = data_concat_arr[:,:,k]
        
        
        
                
        print(i+1, '/', len(date_index))
    
    
    

# ee_dates = ee.List(date_index.values)