#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:53:17 2022

@author: Mitchell
"""

import ee
import pandas as pd
import numpy as np

ee.Initialize()






start = '2019-01-01'
end = '2022-01-01'

# Set size of reduction ('30m' or '10km')
REDUCTION_MODE ='10km'


date_index = pd.date_range(start=start, end=end, freq = '10D')
date_eecoll = ee.ImageCollection([ee.Image(0).set('Date',ee.Date(str(d))) for d in date_index.values[:-1]])

nan_value = -99999

# number of points to sample
NPOINTS = 1000

# input data sources:
   
S1_raw = ee.ImageCollection("COPERNICUS/S1_GRD")
S2_raw = ee.ImageCollection("COPERNICUS/S2")
SMAP = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
TPI = ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI")
LC = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019').select('landcover')
FC = ee.Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01").select('b0').rename(['fc'])

# large USA rough geometry

USA_rough = ee.Geometry.Polygon(
        [[[-121.655078125, 47.97134098275842],
          [-121.74296875, 46.781054468853156],
          [-123.14921875, 47.735432559708755],
          [-123.940234375, 41.947017182140975],
          [-123.6765625, 39.482619912431964],
          [-121.39140625, 39.14261588756164],
          [-118.93046875, 36.15094652424817],
          [-116.030078125, 34.791010541189266],
          [-114.448046875, 32.8931804696908],
          [-112.25078125, 32.077672620403696],
          [-109.174609375, 31.704554516212806],
          [-107.065234375, 32.15211454306765],
          [-105.834765625, 31.85398314718228],
          [-103.6375, 30.348923972724645],
          [-102.143359375, 30.500499250267186],
          [-100.64921875, 29.74027849146342],
          [-98.276171875, 26.56370205472138],
          [-97.48515625, 28.820356772244477],
          [-95.903125, 30.19711355160679],
          [-93.00273437500002, 30.348923972724645],
          [-90.80546875000002, 30.348923972724645],
          [-87.46562500000002, 30.87840414925652],
          [-84.03789062500002, 30.802941602504095],
          [-82.28007812500002, 28.974250944059378],
          [-82.19218750000002, 27.658944872754184],
          [-81.13750000000002, 25.854017411750895],
          [-80.25859375000002, 26.799296955879758],
          [-81.31328125000002, 28.89733237793513],
          [-82.19218750000002, 31.104435140701003],
          [-81.04960937500002, 33.920329092325375],
          [-77.70976562500002, 35.15111555000208],
          [-78.23710937500002, 36.22188306695233],
          [-77.66124089850146, 37.54950959421074],
          [-78.85234375000002, 38.457661565238446],
          [-76.83085937500002, 41.15775831343562],
          [-74.72148437500002, 42.14282493574501],
          [-72.87578125000002, 43.49655581797445],
          [-70.67851562500002, 43.87788728185957],
          [-68.04179687500002, 45.13147386369119],
          [-68.74492187500002, 46.66054698517956],
          [-69.88750000000002, 45.992942429341674],
          [-70.76640625000002, 44.882913434290394],
          [-73.31523437500002, 44.882913434290394],
          [-74.19414062500002, 43.87788728185957],
          [-78.14921875000002, 42.79115532851112],
          [-81.75273437500002, 41.09155203364107],
          [-83.68632812500002, 41.09155203364107],
          [-83.59843750000002, 42.402962806931775],
          [-83.77421875000002, 43.49655581797445],
          [-83.77421875000002, 44.00445896189808],
          [-84.03789062500002, 45.25534961427652],
          [-85.44414062500002, 44.382556961669295],
          [-85.53203125000002, 41.88161385188517],
          [-87.11406250000002, 41.5535923604995],
          [-88.60820312500002, 41.5535923604995],
          [-88.60820312500002, 42.273028083275705],
          [-88.43242187500002, 43.751046164029184],
          [-87.99296875000002, 46.17582452099684],
          [-88.87187500000002, 46.236650501743675],
          [-92.91484375000002, 46.17582452099684],
          [-91.68437500000002, 47.676287727345745],
          [-95.5515625, 48.672633455050736],
          [-97.13359375, 48.61456053275201],
          [-108.471484375, 48.672633455050736],
          [-115.854296875, 48.61456053275201],
          [-121.303515625, 48.672633455050736]]])


USA_box =  ee.Geometry.Rectangle(
        [-126.1375,
        24.231635485098757,
        -66.02031250000002,
        49.855479489102024]
          , None, False)     

# PROJECTIONS 
crs = 'EPSG:4326'
transform = [0.140625,0,-179.9993125,0,-0.09375,90.000125]

smap_crs = 'EPSG:4326'
smap_transform = [0.140625,0,-179.9993125,0,-0.09375,90.000125]
 

s1_crs = 'EPSG:32615'
s1_scale = 10

s2_crs = 'EPSG:32615'
s2_scale = 20

if REDUCTION_MODE == '40m':
    REDUCTION_TRANSFORM = [0.0003593261136478086, 0,0,0,-0.0003593261136478086,0]
    REDUCTION_SCALE = None
    BUFFER_SCALE = 20
elif REDUCTION_MODE == '10km':
    REDUCTION_TRANSFORM = transform
    REDUCTION_SCALE = None
    BUFFER_SCALE = 5000
else:
    raise ValueError()
    
print(REDUCTION_TRANSFORM, REDUCTION_SCALE)

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

s2_btypes = S2_raw.select(s2_bands).filterDate(start,end).filterBounds(USA_rough) \
        .first().bandTypes().getInfo()
        
for b in s2_btypes.keys():
    s2_btypes[b] = ee.PixelType('double', -99999,65535)

def s2CastTypes(s2_image):
    s2_cast = s2_image.cast(s2_btypes)
    return s2_cast
    

# def ndvi(image):
#     ndvi = image.normalizedDifference('B')
    

s2_filtered = S2_raw.filterDate(start,end).filterBounds(USA_box) \
                            .select(s2_bands).map(mask_clouds).map(s2CastTypes)

s1_filtered = S1_raw.filterDate(start,end) \
                .filterBounds(USA_box) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                
tpi_reduce = TPI.reduceResolution(reducer = ee.Reducer.mean(),
                                 bestEffort = True, 
                                 maxPixels = 65000) #\
                           #.reproject(crs = crs, crsTransform = REDUCTION_TRANSFORM, scale = REDUCTION_SCALE)    
                            
lc_reduce = LC.reduceResolution(reducer = ee.Reducer.mode(minBucketWidth = 1, maxBuckets = 85),
                                 bestEffort = True, 
                                 maxPixels = 65000)# \
                          #  .reproject(crs = crs, crsTransform = REDUCTION_TRANSFORM, scale = REDUCTION_SCALE)
                            
fc_reduce = FC.reduceResolution(reducer = ee.Reducer.mean(),
                                 bestEffort = True, 
                                 maxPixels = 65000) 
             

                            
# create netcdf file

def mappingDatesShell(geom):
    def mappingDates(startDate):
        '''
        maps over ee.List of dates
    
        Parameters
        ----------
        startDate : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        INTERVAL = 10
        start = ee.Date(ee.Image(startDate).get('Date'))
        end = start.advance(INTERVAL,'days')
        s1_mean = s1_filtered.filterDate(start,end).filterBounds(geom).mean()
        s2_mean = s2_filtered.filterDate(start,end).filterBounds(geom).mean()
            
                            #.map(mask_clouds).map(s2CastTypes).mean()
        smap_mean = SMAP.select('ssm').filterDate(start,end).filterBounds(geom).mean() \
                .setDefaultProjection(crs = smap_crs, crsTransform = smap_transform)
            # Reduce Resolution to smap
        s1_reduce = s1_mean.setDefaultProjection(crs = s1_crs, scale = s1_scale) \
                        .reduceResolution(reducer = ee.Reducer.mean(),
                            bestEffort = True, 
                            maxPixels = 65000) #\
                       # .reproject(crs = crs, crsTransform = REDUCTION_TRANSFORM, scale = REDUCTION_SCALE)
                                
        s2_1_reduce = s2_mean\
                        .setDefaultProjection(crs = s2_crs, scale = s2_scale) \
                        .reduceResolution(reducer = ee.Reducer.mean(),
                                 bestEffort = True, 
                                 maxPixels = 65000)  #\
                        #    .reproject(crs = crs, crsTransform = REDUCTION_TRANSFORM, scale = REDUCTION_SCALE)
                            
        # s2_2_reduce = s2_mean.select(s2_bands[7:]) \
        #                 .setDefaultProjection(crs = s2_crs, scale = s2_scale)\
        #                 .reduceResolution(reducer = ee.Reducer.mean(),
        #                          bestEffort = True, 
        #                          maxPixels = 65000) #\
                          #  .reproject(crs = crs, crsTransform = REDUCTION_TRANSFORM, scale = REDUCTION_SCALE)
                            
        #\
                          #  .reproject(crs = crs, crsTransform = REDUCTION_TRANSFORM, scale = REDUCTION_SCALE)    
                            
                            
        final_image = s2_1_reduce.addBands(s1_reduce).addBands(smap_mean) \
                                .addBands(tpi_reduce).addBands(lc_reduce).addBands(fc_reduce) \
                                .set('start', start, 'end',end)
                            
        return ee.Image(final_image)
        # return ee.Image(s1_reduce)
    
    return mappingDates


def mapOverImagesShell(geom):
    def mapOverImages(image):     
        reduction = image.reduceRegion(geometry = geom,
            reducer = ee.Reducer.mean(), crs = smap_crs, 
                    crsTransform = REDUCTION_TRANSFORM, scale = REDUCTION_SCALE)
        output_dict = reduction.set('StartDate', ee.Date(image.get('start')))\
                          .set('EndDate', ee.Date(image.get('end')))
        return ee.Feature(None,output_dict )
    return mapOverImages

#####
def mapOverGeoms(feature):
    '''
    takes a feature, returns a 
    '''
    geom = ee.Feature(feature).geometry()
    geom_buffer = geom.buffer(BUFFER_SCALE).bounds()

    
    dateIC = ee.ImageCollection(date_eelist.map( mappingDatesShell(geom_buffer) ) ) 
    
    #
    # AGHAHGHADGHSDKHGSD 
    
    
    # FIX THISSSS BELOW
    
    
    
    geom_coll = ee.FeatureCollection(dateIC.map( mapOverImagesShell(geom_buffer) ) )
    def addCoords(feature):
        return feature.set('PointIndex', feature.get('system:index'))\
            .set('coordinates', geom.coordinates())
    
    return geom_coll.map(addCoords)





#### TRAINING DATA 1

# feature collection with a bunch of points to sample
pixelSamples = ee.FeatureCollection(SMAP.filterBounds(USA_rough).first()
      .sampleRegions(collection = USA_rough, projection = SMAP.first().projection(), geometries = True))

fc = ee.FeatureCollection(pixelSamples.randomColumn(seed = 12).sort('random').toList(NPOINTS))

output_fc = ee.FeatureCollection(fc.map(mapOverGeoms)).flatten()
print(output_fc.first().getInfo())

task = ee.batch.Export.table.toDrive(collection = output_fc, 
                                      description = 'PointSamples{}pts'.format(NPOINTS), 
                                      folder = 'MLEnvironmentGEE', 
                                      fileNamePrefix = 'Point100Sample',
                                      fileFormat = 'CSV')
task.start()


####




#### TRAINING DATA 2: Gauge Stations
# asset_id = 'users/mlt2177/SoilMoistureDownscale/GaugePoints'
# gauge_info_file = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/InSitu/InSituDownload1/namesandcoordinates.csv'
# gauge_df = pd.read_csv(gauge_info_file)
# features = []
# for i in range(len(gauge_df)):
#     g = gauge_df.iloc[i]
#     geom = ee.Geometry.Point([ g.Longitude, g.Latitude,])
#     ft  = ee.Feature(geom, {'system:index':g.GaugeID})
#     features.append(ft)
# fc = ee.FeatureCollection(features)
# print(fc.size().getInfo())
# # 'users/mlt2177/SoilMoistureDownscale/GaugePoints'
# # task = ee.batch.Export.table.toAsset(collection = fc, 
# #                                          description = 'gaugepoints', 
# #                                          assetId = 'users/mlt2177/SoilMoistureDownscale/GaugePoints'
# #                                          )
# # task.start()
# fc = ee.FeatureCollection(asset_id)

# # print(fc.first().getInfo())
# output_fc = ee.FeatureCollection(fc.map(mapOverGeoms)).flatten()

# task = ee.batch.Export.table.toDrive(collection = output_fc, 
#                                       description = 'SMGaugePoints', 
#                                       folder = 'MLEnvironmentGEE', 
#                                       fileNamePrefix = 'SMGaugePoints2',
#                                       fileFormat = 'CSV')
# task.start()



    



# feature = fc.first()
# geom = ee.Feature(feature).geometry()

# dateIC = ee.ImageCollection(date_eelist.map( mappingDatesShell(geom) ) ) 

# geom_coll = ee.FeatureCollection(dateIC.map( mapOverImagesShell(geom) ) )
# print(geom_coll.size().getInfo())
# geom_coll.first().getInfo()

    
