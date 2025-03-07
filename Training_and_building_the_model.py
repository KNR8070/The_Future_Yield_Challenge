#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:32:11 2024

@author: knreddy
"""

#%% Loading modules

import pandas as pd 

import numpy as np 

import tensorflow as tf

from tensorflow import keras 

from tensorflow.keras import layers 

from sklearn.model_selection import train_test_split
#%% loading training data
wrk_dir = '/Volumes/Seagate Expansion Drive/The_Future_Yield/'

train_wheat_tasmin = pd.read_parquet(wrk_dir+'tasmin_wheat_train.parquet')
train_wheat_tasmax = pd.read_parquet(wrk_dir+'tasmax_wheat_train.parquet')
train_wheat_tas = pd.read_parquet(wrk_dir+'tas_wheat_train.parquet')
train_wheat_rsds = pd.read_parquet(wrk_dir+'rsds_wheat_train.parquet')
train_wheat_pr = pd.read_parquet(wrk_dir+'pr_wheat_train.parquet')
train_wheat_soil_co2 = pd.read_parquet(wrk_dir+'soil_co2_wheat_train.parquet')
train_yield = pd.read_parquet(wrk_dir+'train_solutions_wheat.parquet')
#%% Finding a model for a single location first
lat = train_wheat_pr['lat']
lon = train_wheat_pr['lon']
year = train_wheat_soil_co2['real_year']

location1 = [lat.iloc[0],lon.iloc[0]]
lat_loc1 = np.array(lat==lat.iloc[0])
lon_loc1 = np.array(lon==lon.iloc[0])


tasmin_loc1 = np.array(train_wheat_tasmin.iloc[((lat_loc1) & (lon_loc1)),5:])
tasmax_loc1 = np.array(train_wheat_tasmax.iloc[((lat_loc1) & (lon_loc1)),5:])
tas_loc1 = np.array(train_wheat_tas.iloc[((lat_loc1) & (lon_loc1)),5:])
pr_loc1 = np.array(train_wheat_pr.iloc[((lat_loc1) & (lon_loc1)),5:])
rsds_loc1 = np.array(train_wheat_rsds.iloc[((lat_loc1) & (lon_loc1)),5:])
soil_co2_loc1 = np.array(train_wheat_soil_co2.iloc[((lat_loc1) & (lon_loc1)),4:])
nitrogen_loc1 = np.transpose(np.tile(soil_co2_loc1[:,3],[pr_loc1.shape[1],1]))
co2_loc1 = np.transpose(np.tile(soil_co2_loc1[:,2],[pr_loc1.shape[1],1]))
texture_loc1 = np.transpose(np.tile(soil_co2_loc1[:,0],[pr_loc1.shape[1],1]))

yield_loc1  = np.array(train_yield.iloc[((lat_loc1) & (lon_loc1)),5:])

x=tf.stack((tas_loc1,tasmin_loc1,tasmax_loc1,pr_loc1,
                   rsds_loc1,texture_loc1,co2_loc1,nitrogen_loc1),axis=2)
y=yield_loc1
#%% Running the neural network model
# Defining the model 

model = keras.Sequential([ 

    keras.layers.Dense(4, activation='relu'), 

    keras.layers.Dense(2, activation='sigmoid') 
]) 

  
# Compiling the model 

model.compile(optimizer='adam', 

              loss=keras.losses.SparseCategoricalCrossentropy(), 

              metrics=['accuracy']) 

  
# fitting the model 

model.fit(x, y, epochs=10, batch_size=37)