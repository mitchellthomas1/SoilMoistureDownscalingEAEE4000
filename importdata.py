#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:53:50 2022

@author: Mitchell
"""

import numpy as np
import pandas as pd
import os

path = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/'

# files = ['/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/regionsamples/Region{}Samples.csv'.format(i) for i in range(3,11)]
       
def process_gee_file(file):
    # print(file)
    # global df_raw
    df_raw = pd.read_csv(file)
    
    dates = pd.to_datetime(df_raw['start'])
    bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6',
           'B7', 'B8', 'B8A', 'B9', 'VH', 'VV', 'angle', 'ssm']
    df = df_raw[bands]
    df.index = dates
    nan_arr = np.empty((7,7))
    nan_arr[:] = np.nan
    # df.fillna(nan_arr)
    
    # goal 4d array with dimensions: bands, timesteps, lat, lon
    final_arr = np.zeros((len(bands),len(dates), 7,7))
    print('--- NULL Band / Date -----')
    for i, b in enumerate(bands):
        for j, t in enumerate(df.index):
            val = df.loc[t,b]
            if type(val) == str:
                
                final_arr[i,j,:,:] = np.array(eval(val))
    
                
            else:
                print(b, t)
                try:
                    new_val = np.nanmean( (np.array(eval(df.iloc[j-1,i])) , np.array(eval(df.iloc[j+1,i]))), axis = 0)
                    final_arr[i,j,:,:] = new_val
                    print('successfully filled')
                except :
                    final_arr[i,j,:,:] = nan_arr
                    print('not filled')
    print('--------------------------')
    
    return final_arr 

# file = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/pointsamples/Point100Sample.csv'
def process_gee_point_file(file, predictors, fillna = False, si_index = 1):
    
    if 'NDVI' in predictors:
        predictors.remove('NDVI')
    
    # print(file)
    # global df_raw
    df_raw = pd.read_csv(file)
    def unique_preserve_order(li): # get unique list and preserves order
        _, idx = np.unique(li, return_index=True)
        return li[[np.sort(idx)]]
    index_col = np.array([x.split('_')[si_index] for x in df_raw['system:index'].values])
    
    sys_index = unique_preserve_order(index_col)
    
    df_raw['index_column'] = index_col
    df_raw['DateIndex'] = pd.to_datetime(df_raw['StartDate'])
    
    index_coord_df = df_raw[['index_column','coordinates']].drop_duplicates()
    index_coord_dict = {row[0]: {'str_coord' : row[1],
                                 'longitude': eval(row[1])[0],
                                 'latitude':eval(row[1])[1]} for row in index_coord_df.values}
    index_arrays = [df_raw['index_column'], df_raw['DateIndex']]
    index = pd.MultiIndex.from_arrays(index_arrays, names=('PointIndex', 'Date'))
    
    df = df_raw[predictors]
    if fillna == True:
        df = df.fillna(method = 'ffill')
    df.index = index
    

        
    
        

    
    return df, index_coord_dict
# df, coord_dict = process_gee_point_file(file)

def return_all_regions():
    files = ['/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/regionsamples/Region{}Samples.csv'.format(i) for i in range(3,11)]
    regions_arr = np.concatenate([process_gee_file(f) for f in files],axis = 1)
    return regions_arr

# regions_arr = return_all_regions()
# test_stm = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/InSitu/testgauge.stm'
# test_stm2 = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/InSitu/InSituDownload1/TxSON/CR200-14/TxSON_TxSON_CR200-14_sm_0.050000_0.050000_CS655_20190101_20220101.stm'
def stm_to_series(stmfile):
    '''
    Function to take the stm file from ismn and transform it to 
    a pandas series

    Parameters
    ----------
    stmfile : TYPE
        DESCRIPTION.

    Returns
    -------
    resampled : TYPE
        DESCRIPTION.
    tuple
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    '''
    df = pd.read_csv(stmfile, delimiter = ' ').iloc[:,0:3]
    df.columns = ['Date','time','sm']
    df = df[df.sm != -9999]
    sm_series = df['sm']

    # dates
    date_li = [ str(df['Date'].values[i] + '-' +  df['time'].values[i]) for i in range(len(sm_series)) ]

    date_index = pd.to_datetime(  date_li   ,
                   format = '%Y/%m/%d-%H:%M')

    sm_series.index = date_index
    # sm_series.
    hourly_3yr = pd.date_range(start=pd.Timestamp(2019,1,1), 
                                   end=pd.Timestamp(2022,1,1), freq='H')
    series_hour3yr = sm_series.reindex(hourly_3yr)
    resampled = series_hour3yr.resample('10D').mean()
    
    #get latitude and longitude
    with open(stmfile) as f:
        first_line = f.readline()
        attributes = [x for x in first_line.split(' ') if x != '' ]
        lat = float(attributes[3])
        lon = float(attributes[4])
        name = '-'.join(attributes[1:3])
    
    
    return resampled, (lon, lat), name

   
def import_ismn(ismn_download, start, end):
    #minimum datapoints to keep a series
    MIN_POINTS = 80
    # files = os.walk(ismn_download) 
    systems = [x for x in os.listdir(ismn_download) if x not in ['.DS_Store','Readme.txt','Metadata.xml','ISMN_qualityflags_description.txt', 'namesandcoordinates.csv']]
    i = 1
    # maps gauges to 
    gauge_dict = {}
    all_series = []
    # default_index = pd.date_range(start=pd.Timestamp(2019,1,1), end=pd.Timestamp(2022,1,1), freq='10D')
    coord_list = []
    for sys in systems:
        # print(sys)
        system_dir = ismn_download + '/' + sys
        for site in [f for f in os.listdir(system_dir) if f != '.DS_Store']:
            print(sys, site)
            site_file_dir = system_dir + '/' + site
            site_files = [f for f in os.listdir(site_file_dir) if '.stm' in f]
            start_depths = np.array([float(f.split('_')[4]) for f in site_files ])
            min_files = np.where(start_depths == min(start_depths))[0]

            if len(min_files) == 0:
                raise Exception('min files length 0')
            elif len(min_files) == 1:
                
                file = ismn_download + '/' + sys + '/' + site + '/' + site_files[min_files[0]]
                series, foo1, foo2 = stm_to_series(file)

                
            else:
                # print(sys, site, 'Multiple!!')
                prefix = ismn_download + '/' + sys + '/' + site + '/'
                # print(prefix)
                series = pd.DataFrame([stm_to_series(prefix + site_files[i])[0] for i in min_files]).mean()
    
            if series.dropna().size >= MIN_POINTS:
                all_series.append(series[:-1])
                i += 1
                prefix = ismn_download + '/' + sys + '/' + site + '/'
                _ , (lon, lat), name = stm_to_series(prefix + site_files[min_files[0]])
                # print(name)
                net, station = name.split('-')
                gauge_dict['gauge_{}'.format(i)] = {'lon': lon,'lat':lat,
                                                'network': net, 'stationname': station}
                coord_list.append([lon, lat])
   
        
    # index = pd.date_range(start=pd.Timestamp(2019,1,1), end=pd.Timestamp(2022,1,1), freq='10D')
    final_df = pd.concat(all_series, axis = 0)

    # reorder into vertically concatted array 
    # if coordinate_dict is not None:
    #     vertical_li = []
    #     sat_coord_list = [eval(coordinate_dict[x]['str_coord']) for x in coordinate_dict]
    #     sat_coord_arr = np.round(np.array(sat_coord_list).astype(float), decimals = 4)
    #     ismn_coord_arr = np.round(np.array(coord_list).astype(float), decimals = 4)
    #     # create empty placeholder
    #     empty = np.empty(len(final_df_wide.index))
    #     empty[:] = np.nan
    #     empty_series = pd.Series(empty, index= final_df_wide.index)
    #     # go through each satellite coordinate, match up with 
    #     for gauge in sat_coord_arr:
    #         where = np.where(ismn_coord_arr == gauge)
    #         print(where)
            
    
    # final_df_wide.columns = list(range(1,i))
    return final_df, gauge_dict
        
# start = pd.Timestamp(2019,1,1)
# end = pd.Timestamp(2022,1,1)
# ismn_download = '/Users/Mitchell/Documents/MLEnvironment/SoilMoistureDownscalingEAEE4000/DataDownload/InSitu/InSituDownload1'

# final_df, gauge_dict = import_ismn(ismn_download, start, end)


# final_df.to_csv(path+'DataDownload/InSitu/SoilMoistureDataFrameGreaterThan80_2.csv')

def download_gauge_csv(gauge_dict):
    download_file = path + 'DataDownload/InSitu/InSituDownload1/namesandcoordinates.csv'
    gauge_ids = []
    networks = []
    stations = []
    lats = []
    lons = []
    for key in gauge_dict.keys():
        gd = gauge_dict[key]
        gauge_ids.append(key)
        networks.append(gd['network'])
        stations.append(gd['stationname'])
        lats.append(gd['lat'])
        lons.append(gd['lon'])
        
    g_df = pd.DataFrame({'GaugeID': gauge_ids,
                  'Network': networks,
                  'Station':stations,
                  'Latitude':lats,
                  'Longitude': lons})
    g_df.to_csv(download_file)
    

        
    
    # print(root)
    # for dir in root :
    #     for files in dir:
    #         print(files)
