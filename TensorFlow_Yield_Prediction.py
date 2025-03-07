#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:31:34 2024

@author: knreddy
"""
#%% loading modules
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
#%% loading training data
wrk_dir = '/Volumes/Seagate Expansion Drive/The_Future_Yield/'

train_wheat_tasmin = pd.read_parquet(wrk_dir+'tasmin_wheat_train.parquet')
train_wheat_tasmax = pd.read_parquet(wrk_dir+'tasmax_wheat_train.parquet')
train_wheat_tas = pd.read_parquet(wrk_dir+'tas_wheat_train.parquet')
train_wheat_rsds = pd.read_parquet(wrk_dir+'rsds_wheat_train.parquet')
train_wheat_pr = pd.read_parquet(wrk_dir+'pr_wheat_train.parquet')
train_wheat_soil_co2 = pd.read_parquet(wrk_dir+'soil_co2_wheat_train.parquet')
train_yield = pd.read_parquet(wrk_dir+'train_solutions_wheat.parquet')

eval_wheat_tasmin = pd.read_parquet(wrk_dir+'tasmin_wheat_test.parquet')
eval_wheat_tasmax = pd.read_parquet(wrk_dir+'tasmax_wheat_test.parquet')
eval_wheat_tas = pd.read_parquet(wrk_dir+'tas_wheat_test.parquet')
eval_wheat_rsds = pd.read_parquet(wrk_dir+'rsds_wheat_test.parquet')
eval_wheat_pr = pd.read_parquet(wrk_dir+'pr_wheat_test.parquet')
eval_wheat_soil_co2 = pd.read_parquet(wrk_dir+'soil_co2_wheat_test.parquet')
#%% extracting pre-season climate and season climate data
wrk_dir = '/Volumes/Seagate Expansion Drive/The_Future_Yield/'

data_usedfor = ['train','test']
crops = ['wheat','maize']
meteo_var1 = ['tas','tasmin','tasmax','rsds','pr','soil_co2']
ann_var = ['co2','nitrogen','texture','year','lat','lon']
for data_purpose in data_usedfor:
    print('extracting '+data_purpose+'ing data' )
    for crop in crops:
        print('extracting '+crop+' data' )
        for var in meteo_var1:
            if meteo_var1.index(var) == 5:
                print('extracting final variable')
            input_data = wrk_dir+var+'_'+crop+'_'+data_purpose+'.parquet'
            out_put = data_purpose+'_'+crop+'_'+var
            globals()[out_put] = pd.read_parquet(input_data)
            if var == 'soil_co2':
                for var1 in ann_var:
                    out_put1 = data_purpose+'_'+crop+'_'+var1
                    if var1 == 'year':
                        globals()[out_put1] = pd.array(eval(out_put)['real_year'])
                    elif var1== 'texture':
                        globals()[out_put1] = pd.array(eval(out_put)['texture_class'])
                    else:
                        globals()[out_put1] = pd.array(eval(out_put)[var1])            
            elif var == 'pr':
                out_put_preseaon_sum = out_put+'_preseason_sum'
                out_put_seaon_sum = out_put+'_season_sum'
                out_put_preseaon_ndays = out_put+'_preseason_ndays'
                out_put_seaon_ndays = out_put+'_season_ndays'
                eval(out_put)[eval(out_put) == 0] = np.nan
                globals()[out_put_preseaon_sum] = pd.array((eval(out_put).iloc[:,5:36]).sum(axis=1,skipna=True))
                globals()[out_put_seaon_sum] = pd.array((eval(out_put).iloc[:,36:]).sum(axis=1,skipna=True))
                globals()[out_put_preseaon_ndays] = pd.array((eval(out_put).iloc[:,5:36]).sum(axis=1,skipna=True))/pd.array((eval(out_put).iloc[:,5:36]).mean(axis=1,skipna=True))
                globals()[out_put_seaon_ndays] = pd.array((eval(out_put).iloc[:,36:]).sum(axis=1,skipna=True))/pd.array((eval(out_put).iloc[:,36:]).mean(axis=1,skipna=True))            
            else:                    
                out_put_preseaon_mean = out_put+'_preseason_mean'
                out_put_seaon_mean = out_put+'_season_mean'
                out_put_preseaon_std = out_put+'_preseason_std'
                out_put_seaon_std = out_put+'_season_std'
                globals()[out_put_preseaon_mean] = pd.array((eval(out_put).iloc[:,5:36]).mean(axis=1))
                globals()[out_put_seaon_mean] = pd.array((eval(out_put).iloc[:,36:]).mean(axis=1))
                globals()[out_put_preseaon_std] = pd.array((eval(out_put).iloc[:,5:36]).std(axis=1))
                globals()[out_put_seaon_std] = pd.array((eval(out_put).iloc[:,36:]).std(axis=1))

train_wheat_yield_data = pd.read_parquet(wrk_dir+'train_solutions_wheat.parquet')
train_wheat_yield = pd.array(train_wheat_yield_data['yield'])

train_maize_yield_data = pd.read_parquet(wrk_dir+'train_solutions_maize.parquet')
train_maize_yield = pd.array(train_maize_yield_data['yield'])
#%% creating dataframe for training
wheat_train_data = pd.DataFrame({'lat':list(train_wheat_lat),
                                 'lon':list(train_wheat_lon),
                                 'texture':list(train_wheat_texture),
                                 'co2':list(train_wheat_co2),
                                 'nitrogen':list(train_wheat_nitrogen),
                                 'tas_preseason_mean':list(train_wheat_tas_preseason_mean),
                                 'tas_preseason_std':list(train_wheat_tas_preseason_std),
                                 'tas_season_mean':list(train_wheat_tas_season_mean),
                                 'tas_season_std':list(train_wheat_tas_season_std),
                                 'tasmin_preseason_mean':list(train_wheat_tasmin_preseason_mean),
                                 'tasmin_preseason_std':list(train_wheat_tasmin_preseason_std),
                                 'tasmin_season_mean':list(train_wheat_tasmin_season_mean),
                                 'tasmin_season_std':list(train_wheat_tasmin_season_std),
                                 'tasmax_preseason_mean':list(train_wheat_tasmax_preseason_mean),
                                 'tasmax_preseason_std':list(train_wheat_tasmax_preseason_std),
                                 'tasmax_season_mean':list(train_wheat_tasmax_season_mean),
                                 'tasmax_season_std':list(train_wheat_tasmax_season_std),
                                 'tasmin_preseason_mean':list(train_wheat_tasmin_preseason_mean),
                                 'tasmin_preseason_std':list(train_wheat_tasmin_preseason_std),
                                 'tasmin_season_mean':list(train_wheat_tasmin_season_mean),
                                 'tasmin_season_std':list(train_wheat_tasmin_season_std),
                                 'rsds_preseason_mean':list(train_wheat_rsds_preseason_mean),
                                 'rsds_preseason_std':list(train_wheat_rsds_preseason_std),
                                 'rsds_season_mean':list(train_wheat_rsds_season_mean),
                                 'rsds_season_std':list(train_wheat_rsds_season_std),
                                 'pr_preseason_sum':list(train_wheat_pr_preseason_sum),
                                 'pr_preseason_ndays':list(train_wheat_pr_preseason_ndays),
                                 'pr_season_sum':list(train_wheat_pr_season_sum),
                                 'pr_season_ndays':list(train_wheat_pr_season_ndays),
                                       })

wheat_test_data = pd.DataFrame({'lat':list(test_wheat_lat),
                                 'lon':list(test_wheat_lon),
                                 'texture':list(test_wheat_texture),
                                 'co2':list(test_wheat_co2),
                                 'nitrogen':list(test_wheat_nitrogen),
                                 'tas_preseason_mean':list(test_wheat_tas_preseason_mean),
                                 'tas_preseason_std':list(test_wheat_tas_preseason_std),
                                 'tas_season_mean':list(test_wheat_tas_season_mean),
                                 'tas_season_std':list(test_wheat_tas_season_std),
                                 'tasmin_preseason_mean':list(test_wheat_tasmin_preseason_mean),
                                 'tasmin_preseason_std':list(test_wheat_tasmin_preseason_std),
                                 'tasmin_season_mean':list(test_wheat_tasmin_season_mean),
                                 'tasmin_season_std':list(test_wheat_tasmin_season_std),
                                 'tasmax_preseason_mean':list(test_wheat_tasmax_preseason_mean),
                                 'tasmax_preseason_std':list(test_wheat_tasmax_preseason_std),
                                 'tasmax_season_mean':list(test_wheat_tasmax_season_mean),
                                 'tasmax_season_std':list(test_wheat_tasmax_season_std),
                                 'tasmin_preseason_mean':list(test_wheat_tasmin_preseason_mean),
                                 'tasmin_preseason_std':list(test_wheat_tasmin_preseason_std),
                                 'tasmin_season_mean':list(test_wheat_tasmin_season_mean),
                                 'tasmin_season_std':list(test_wheat_tasmin_season_std),
                                 'rsds_preseason_mean':list(test_wheat_rsds_preseason_mean),
                                 'rsds_preseason_std':list(test_wheat_rsds_preseason_std),
                                 'rsds_season_mean':list(test_wheat_rsds_season_mean),
                                 'rsds_season_std':list(test_wheat_rsds_season_std),
                                 'pr_preseason_sum':list(test_wheat_pr_preseason_sum),
                                 'pr_preseason_ndays':list(test_wheat_pr_preseason_ndays),
                                 'pr_season_sum':list(test_wheat_pr_season_sum),
                                 'pr_season_ndays':list(test_wheat_pr_season_ndays),
                                       })

maize_train_data = pd.DataFrame({'lat':list(train_maize_lat),
                                 'lon':list(train_maize_lon),
                                 'texture':list(train_maize_texture),
                                 'co2':list(train_maize_co2),
                                 'nitrogen':list(train_maize_nitrogen),
                                 'tas_preseason_mean':list(train_maize_tas_preseason_mean),
                                 'tas_preseason_std':list(train_maize_tas_preseason_std),
                                 'tas_season_mean':list(train_maize_tas_season_mean),
                                 'tas_season_std':list(train_maize_tas_season_std),
                                 'tasmin_preseason_mean':list(train_maize_tasmin_preseason_mean),
                                 'tasmin_preseason_std':list(train_maize_tasmin_preseason_std),
                                 'tasmin_season_mean':list(train_maize_tasmin_season_mean),
                                 'tasmin_season_std':list(train_maize_tasmin_season_std),
                                 'tasmax_preseason_mean':list(train_maize_tasmax_preseason_mean),
                                 'tasmax_preseason_std':list(train_maize_tasmax_preseason_std),
                                 'tasmax_season_mean':list(train_maize_tasmax_season_mean),
                                 'tasmax_season_std':list(train_maize_tasmax_season_std),
                                 'tasmin_preseason_mean':list(train_maize_tasmin_preseason_mean),
                                 'tasmin_preseason_std':list(train_maize_tasmin_preseason_std),
                                 'tasmin_season_mean':list(train_maize_tasmin_season_mean),
                                 'tasmin_season_std':list(train_maize_tasmin_season_std),
                                 'rsds_preseason_mean':list(train_maize_rsds_preseason_mean),
                                 'rsds_preseason_std':list(train_maize_rsds_preseason_std),
                                 'rsds_season_mean':list(train_maize_rsds_season_mean),
                                 'rsds_season_std':list(train_maize_rsds_season_std),
                                 'pr_preseason_sum':list(train_maize_pr_preseason_sum),
                                 'pr_preseason_ndays':list(train_maize_pr_preseason_ndays),
                                 'pr_season_sum':list(train_maize_pr_season_sum),
                                 'pr_season_ndays':list(train_maize_pr_season_ndays),
                                       })

maize_test_data = pd.DataFrame({'lat':list(test_maize_lat),
                                 'lon':list(test_maize_lon),
                                 'texture':list(test_maize_texture),
                                 'co2':list(test_maize_co2),
                                 'nitrogen':list(test_maize_nitrogen),
                                 'tas_preseason_mean':list(test_maize_tas_preseason_mean),
                                 'tas_preseason_std':list(test_maize_tas_preseason_std),
                                 'tas_season_mean':list(test_maize_tas_season_mean),
                                 'tas_season_std':list(test_maize_tas_season_std),
                                 'tasmin_preseason_mean':list(test_maize_tasmin_preseason_mean),
                                 'tasmin_preseason_std':list(test_maize_tasmin_preseason_std),
                                 'tasmin_season_mean':list(test_maize_tasmin_season_mean),
                                 'tasmin_season_std':list(test_maize_tasmin_season_std),
                                 'tasmax_preseason_mean':list(test_maize_tasmax_preseason_mean),
                                 'tasmax_preseason_std':list(test_maize_tasmax_preseason_std),
                                 'tasmax_season_mean':list(test_maize_tasmax_season_mean),
                                 'tasmax_season_std':list(test_maize_tasmax_season_std),
                                 'tasmin_preseason_mean':list(test_maize_tasmin_preseason_mean),
                                 'tasmin_preseason_std':list(test_maize_tasmin_preseason_std),
                                 'tasmin_season_mean':list(test_maize_tasmin_season_mean),
                                 'tasmin_season_std':list(test_maize_tasmin_season_std),
                                 'rsds_preseason_mean':list(test_maize_rsds_preseason_mean),
                                 'rsds_preseason_std':list(test_maize_rsds_preseason_std),
                                 'rsds_season_mean':list(test_maize_rsds_season_mean),
                                 'rsds_season_std':list(test_maize_rsds_season_std),
                                 'pr_preseason_sum':list(test_maize_pr_preseason_sum),
                                 'pr_preseason_ndays':list(test_maize_pr_preseason_ndays),
                                 'pr_season_sum':list(test_maize_pr_season_sum),
                                 'pr_season_ndays':list(test_maize_pr_season_ndays),
                                       })

#%% Deleting variables

data_usedfor = ['train','test']
crops = ['wheat','maize']
all_var = ['tas','tasmin','tasmax','rsds','pr','co2','nitrogen','texture','year','lat','lon']

for data_purpose in data_usedfor:
    print('deleting '+data_purpose+'ing data' )
    for crop in crops:
        print('deleting '+crop+' data' )
        for var in all_var:
            if var == ('co2') or var == ('nitrogen') or var == ('texture') or var == ('year') or var == ('lat') or var == ('lon'):
                data_name = data_purpose+'_'+crop+'_'+var
                del globals()[data_name]
            else:
                data_name1 = data_purpose+'_'+crop+'_'+var+'_preseason_mean'
                data_name2 = data_purpose+'_'+crop+'_'+var+'_preseason_std'
                data_name3 = data_purpose+'_'+crop+'_'+var+'_season_mean'
                data_name4 = data_purpose+'_'+crop+'_'+var+'_season_std'
                del globals()[data_name1]
                del globals()[data_name2]
                del globals()[data_name3]
                del globals()[data_name4]
#%% Random Forest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
scaler = StandardScaler()
wheat_data_scaled = scaler.fit_transform(wheat_train_data)
maize_data_scaled = scaler.fit_transform(maize_train_data)

# Split Data
X_W_train, X_W_val, y_W_train, y_W_val = train_test_split(wheat_data_scaled, train_wheat_yield, test_size=0.2, random_state=42)
X_M_train, X_M_val, y_M_train, y_M_val = train_test_split(maize_data_scaled, train_maize_yield, test_size=0.2, random_state=42)

# Model Training
wheat_model = RandomForestRegressor()
wheat_model.fit(X_W_train, y_W_train)

maize_model = RandomForestRegressor()
maize_model.fit(X_M_train, y_M_train)

# Model Evaluation
W_y_pred = wheat_model.predict(X_W_val)
rmse = np.sqrt(mean_squared_error(y_W_val, W_y_pred))
print(f"Validation RMSE: {rmse}")

M_y_pred = maize_model.predict(X_M_val)
rmse = np.sqrt(mean_squared_error(y_M_val, M_y_pred))
print(f"Validation RMSE: {rmse}")

wheat_test_data_scaled = scaler.transform(wheat_test_data)
wheat_test_predictions = wheat_model.predict(wheat_test_data_scaled)

maize_test_data_scaled = scaler.transform(maize_test_data)
maize_test_predictions = maize_model.predict(maize_test_data_scaled)

submission_data = pd.read_csv(wrk_dir+'sample_submission.csv')

# Prepare Submission
my_submission = pd.DataFrame({"ID": submission_data["ID"], 
                           "yield": np.concatenate([maize_test_predictions,wheat_test_predictions])})
my_submission.to_csv("/Users/knreddy/Documents/The_Future_Yield/Yield_submission_KNR_v4_19Aug_rainydays.csv", index=False)
#%% Gradient boost method
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Convert the data to DMatrix, which is optimized for XGBoost
W_dtrain = xgb.DMatrix(X_W_train, label=y_W_train)
W_dval = xgb.DMatrix(X_W_val, label=y_W_val)

M_dtrain = xgb.DMatrix(X_M_train, label=y_M_train)
M_dval = xgb.DMatrix(X_M_val, label=y_M_val)

# Set parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train the model
xgb_wheat_model = xgb.train(params, W_dtrain, num_boost_round=100000, 
                            evals=[(W_dtrain, 'train'), (W_dval, 'eval')], 
                            early_stopping_rounds=500, verbose_eval=100)
xgb_maize_model = xgb.train(params, M_dtrain, num_boost_round=100000, 
                            evals=[(M_dtrain, 'train'), (M_dval, 'eval')], 
                            early_stopping_rounds=500, verbose_eval=100)


# Predict and evaluate
W_y_pred = xgb_wheat_model.predict(W_dval)
W_rmse = np.sqrt(mean_squared_error(y_W_val, W_y_pred))
print(f"Validation RMSE: {W_rmse}")

M_y_pred = xgb_wheat_model.predict(M_dval)
M_rmse = np.sqrt(mean_squared_error(y_M_val, M_y_pred))
print(f"Validation RMSE: {M_rmse}")

W_dtest = xgb.DMatrix(wheat_test_data)
xgb_wheat_test_predictions = xgb_wheat_model.predict(W_dtest)

M_dtest = xgb.DMatrix(maize_test_data)
xgb_maize_test_predictions = xgb_maize_model.predict(M_dtest)

# Prepare Submission
my_xgb_submission = pd.DataFrame({"ID": submission_data["ID"], 
                           "yield": np.concatenate([xgb_maize_test_predictions,xgb_wheat_test_predictions])})
my_xgb_submission.to_csv("/Users/knreddy/Documents/The_Future_Yield/Yield_submission_KNR_v3_xgbmethod_100000Nums.csv", index=False)
#%% Neural networks
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
#%%
# Function to create model, required for KerasRegressor
def create_model(learning_rate=0.01, activation='relu', neurons=32, dropout_rate=0.0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neurons, activation=activation, input_shape=(wheat_data_scaled.shape[1],)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
#%%
# Wrap Keras model with KerasRegressor
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the parameter grid
param_dist = {
    'learning_rate': [0.001, 0.01, 0.1],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'neurons': [16, 32, 64, 128],
    'dropout_rate': [0.0, 0.2, 0.4],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 200]
}

# Random search of parameters
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                   n_iter=20, cv=3, verbose=2, n_jobs=-1)

# Fit random search
random_search_result = random_search.fit(wheat_data_scaled, train_wheat_yield)

# Display the best hyperparameters
print(f"Best: {random_search_result.best_score_} using {random_search_result.best_params_}")

# Use the best estimator to make predictions
best_model = random_search_result.best_estimator_
y_test_pred = best_model.predict(X_test_scaled)

# Create a DataFrame for the submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'yield': y_test_pred
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
#%%
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_W_train)
X_val_scaled = scaler.transform(X_W_val)
X_test_scaled = scaler.transform(wheat_test_data)

#%%
# Build the model
from tensorflow.keras import layers, models
W_NN_model = models.Sequential()
W_NN_model.add(layers.Dense(16, activation='relu', input_shape=(X_W_train.shape[1],)))
W_NN_model.add(layers.Dense(8, activation='relu'))
W_NN_model.add(layers.Dense(4, activation='relu'))
W_NN_model.add(layers.Dense(1))

W_NN_model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# Train the model
history = W_NN_model.fit(X_W_train, y_W_train, epochs=10, validation_data=(X_W_val, y_W_val), 
                    batch_size=32, verbose=2)

# Make predictions
y_W_test_pred = model.predict(X_W_test)

#%% To improve speed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Load your dataset
train_df = pd.read_parquet('train_data.parquet')
test_df = pd.read_parquet('test_data.parquet')

# Split features and target
X_train = train_df.drop(columns=['yield'])
y_train = train_df['yield']
X_test = test_df.drop(columns=['yield'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create tf.data.Datasets
def preprocess(features, labels):
    return features, labels

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train))
train_dataset = train_dataset.map(preprocess).shuffle(buffer_size=1024).batch(64).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val))
val_dataset = val_dataset.batch(64).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Build the model
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

model = create_model()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[early_stopping, reduce_lr], verbose=2)

# Make predictions on the test set
y_test_pred = model.predict(X_test_scaled)

# Create a DataFrame for the submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'yield': y_test_pred[:, 0]
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Define the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
#%% Checking the best parameters for wheat
# Perform Randomized Search for hyperparameter tuning
import joblib

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, 
                                   cv=3, verbose=2, n_jobs=-1, random_state=42)
random_search.fit(X_W_train, y_W_train)

# Display the best hyperparameters
print(f"Best: {random_search.best_score_} using {random_search.best_params_}")

# Save the model
best_W_rf = random_search.best_estimator_
joblib.dump(best_W_rf, '/Users/knreddy/Documents/The_Future_Yield/best_Wheat_random_forest_model.pkl')

# # Load the model
# best_W_model = joblib.load('best_Wheat_random_forest_model.pkl')
#%% Checking the best parameters for maize
# Perform Randomized Search for hyperparameter tuning
random_M_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, 
                                   cv=3, verbose=2, n_jobs=-1, random_state=42)
random_M_search.fit(X_M_train, y_M_train)

# Display the best hyperparameters
print(f"Best: {random_search.best_score_} using {random_search.best_params_}")

# Save the model
best_M_rf = random_M_search.best_estimator_
joblib.dump(best_M_rf, '/Users/knreddy/Documents/The_Future_Yield/best_Maize_random_forest_model.pkl')
#%% Wheat model
# Train the best model
best_W_rf = random_search.best_estimator_
best_W_rf.fit(X_W_train, y_W_train)

# Evaluate the model on the validation set
y_W_val_pred = best_W_rf.predict(X_W_val)
val_rmse = mean_squared_error(y_W_val, y_W_val_pred, squared=False)
print(f"Validation RMSE: {val_rmse}")
#%% Maize model evaluation 
# Train the best model
best_M_rf = random_M_search.best_estimator_
best_M_rf.fit(X_M_train, y_M_train)

# Evaluate the model on the validation set
y_M_val_pred = best_M_rf.predict(X_M_val)
M_val_rmse = mean_squared_error(y_M_val, y_M_val_pred, squared=False)
print(f"Validation RMSE: {M_val_rmse}")
#%% Testing data and submission
wheat_test_data_scaled = scaler.transform(wheat_test_data)
maize_test_data_scaled = scaler.transform(maize_test_data)

wheat_test_predictions = best_W_rf.predict(wheat_test_data_scaled)
maize_test_predictions = best_M_rf.predict(maize_test_data_scaled)

submission_data = pd.read_csv(wrk_dir+'sample_submission.csv')

# Prepare Submission
my_submission = pd.DataFrame({"ID": submission_data["ID"], 
                           "yield": np.concatenate([maize_test_predictions,wheat_test_predictions])})
my_submission.to_csv("/Users/knreddy/Documents/The_Future_Yield/Yield_submission_KNR_30June.csv", index=False)
