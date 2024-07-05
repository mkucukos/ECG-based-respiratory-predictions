#!/usr/bin/env python
# coding: utf-8

# # Important Variable

# In[1]:


#!jupyter nbconvert --to script model_build.ipynb

import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.signal import resample

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, linregress
import random
import tensorflow as tf
from tqdm.notebook import tqdm
from tensorflow.python.client import device_lib
import matplotlib.colors as mcolors


# In[2]:


print(tf.__version__)


# In[3]:


tf.config.experimental.list_physical_devices('GPU')
device_lib.list_local_devices()


# In[ ]:





# In[4]:


MODEL_NAME = '3_inputs_model'
RUN = 'run6_' + MODEL_NAME+ '/'

CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/dataset/'
MASTER_OUTPUT_DIR = CURRENT_DIR + '/output/'
OUTPUT_DIR = MASTER_OUTPUT_DIR + RUN
print(DATA_DIR)
print(OUTPUT_DIR)


# In[5]:


#data_df[data_df['subject_id'] != 'f2y08']['snr'].describe()


# # Load data

# In[6]:


data_df = pd.read_pickle(DATA_DIR + 'fantasia_dataset_preprocessing.plk')
data_df


# In[ ]:





# In[ ]:





# # Preprocessing

# In[7]:


chunks_df = data_df #allias


# In[8]:


chunks_df = chunks_df.dropna(subset=['Mean_Respiration_Rate','hrv'])


# ## Standardization

# In[9]:


# Convert 'ecg' column to a 3D numpy array
X = np.array(chunks_df['ecg'].tolist())
Y = chunks_df[['Mean_Respiration_Rate','Total_Area_Trough_to_Peak','Total_Area_Peak_to_Trough',
            'Mean_latency_Trough_to_Peak','Mean_latency_Peak_to_Trough']].values

# Normalize each ECG chunk individually and handle NaNs
X_normalized = np.zeros_like(X)
for i in range(X.shape[0]):
    mean = np.mean(X[i])
    std = np.std(X[i])
    if std == 0:  # Avoid division by zero
        print('sid == 0')
        std = 1
    X_normalized[i] = (X[i] - mean) / std


# In[10]:


X_feature = np.array(chunks_df[['hr_min','hr_mean','hr_max','hrv','snr']])


# Normalize each ECG chunk individually and handle NaNs
X_feature_normalized = np.zeros_like(X_feature)
for i in range(X_feature.shape[1]):
    mean = np.mean(X_feature[:,i])
    std = np.std(X_feature[:,i])
    if std == 0:  # Avoid division by zero
        print('sid == 0')
        std = 1
    X_feature_normalized[:,i] = (X_feature[:,i] - mean) / std


# In[11]:


print(X_normalized.shape)
print(X_feature_normalized.shape)
print(Y.shape)


# In[ ]:





# In[12]:


# Remove Nan value
nan_rows = np.isnan(X_normalized).any(axis=1)

# Remove rows with NaNs
X_normalized = X_normalized[~nan_rows]
X_feature_normalized = X_feature_normalized[~nan_rows]
Y = Y[~nan_rows]
cleaned_chunks_df = chunks_df[~nan_rows]

print(nan_rows.sum())
print(f"NaN in X_normalized after cleaning: {np.isnan(X_normalized).sum()}, Inf in X_normalized: {np.isinf(X_normalized).sum()}")


# In[13]:


print(X_normalized.shape)
print(X_feature_normalized.shape)
print(Y.shape)


# In[14]:


# # Calculate the 2.5th and 97.5th percentiles
# for i in range(Y.shape[1]):
#     lower_percentile = np.percentile(Y[:,i], 0.5)
#     upper_percentile = np.percentile(Y[:,i], 99.5)



#     # Verify the clipping
#     print('\n', i)
#     print(f"1th percentile: {lower_percentile}")
#     print(f"99th percentile: {upper_percentile}")
#     print(f"Before clipping: min = {Y[:,i].min()}, max = {Y[:,i].max()}")
#     # Clip the values in y_train
#     Y[:,i] = np.clip(Y[:,i], lower_percentile, upper_percentile)
#     print(f"After clipping: min = {Y[:,i].min()}, max = {Y[:,i].max()}")

#     # Continue with training the model using y_train_clipped


# Spliting

# In[15]:


from sklearn.model_selection import train_test_split


subjects = cleaned_chunks_df['subject_id'].unique()
train_val_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=96)
train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15, random_state=96)

# Create boolean masks for train, validation, and test subjects
train_mask = cleaned_chunks_df['subject_id'].isin(train_subjects)
val_mask = cleaned_chunks_df['subject_id'].isin(val_subjects)
test_mask = cleaned_chunks_df['subject_id'].isin(test_subjects)

# Use these masks to create the actual train, validation, and test datasets
train_data = cleaned_chunks_df[train_mask]
val_data = cleaned_chunks_df[val_mask]
test_data = cleaned_chunks_df[test_mask]

# Now you have train, validation, and test datasets based on the subjects
test_subject_id = cleaned_chunks_df[test_mask]['subject_id'].to_numpy()


# In[16]:


print(train_subjects)
print(val_subjects)
print(test_subjects)


# In[17]:


X_normalized


# In[18]:


# Apply the masks to create train, validation, and test sets
X_train = np.expand_dims(X_normalized[train_mask], 2)
Y_train = Y[train_mask] 
X_val = np.expand_dims(X_normalized[val_mask], 2)
Y_val = Y[val_mask]
X_test = np.expand_dims(X_normalized[test_mask], 2)
Y_test = Y[test_mask]

# Print shapes of the datasets to verify
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {Y_train.shape}")
print(f"y_val shape: {Y_val.shape}")
print(f"y_test shape: {Y_test.shape}")
print(f"test_subject_id shape: {test_subject_id.shape}")


# In[19]:


# Apply the masks to create train, validation, and test sets
X_train_feature = X_feature_normalized[train_mask]
X_val_feature = X_feature_normalized[val_mask]
X_test_feature = X_feature_normalized[test_mask]


# Print shapes of the datasets to verify
print(f"X_train_feature shape: {X_train_feature.shape}")
print(f"X_val_feature shape: {X_val_feature.shape}")
print(f"X_test_feature shape: {X_test_feature.shape}")


# In[20]:


def compute_freq_domain(X, SAMPLE_RATE = 250, N_SAMPLES = 30*250):
    X_freq = []
    for i in range(X.shape[0]):
        data = X_train[i]
        frequency_domain = np.fft.fft(data)
        #frequencies = np.fft.fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
        X_freq.append(np.abs(frequency_domain))
    X_freq = np.array(X_freq)
    return X_freq
    


# In[21]:


X_train_freq = compute_freq_domain(X_train)
X_val_freq =  compute_freq_domain(X_val)
X_test_freq = compute_freq_domain(X_test)

print(f"X_train_freq shape: {X_train_freq.shape}")
print(f"X_val_freq shape: {X_val_freq.shape}")
print(f"X_test_freq shape: {X_test_freq.shape}")


# In[22]:


import pywt

def extract_wavelet_features(X, wavelet='db4', level=4):
    n_samples = X.shape[0]
    
    #calculate n_feature
    n_features = 0
    temp = pywt.wavedec(np.zeros(X.shape[1]), wavelet, level=level)
    for i in temp:
        n_features += len(i)

    X_features = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        current_signal = X[i, :, 0]
        
        coeffs = pywt.wavedec(current_signal, wavelet, level=level)
        features = np.concatenate(coeffs)
        
        X_features[i, :] = features
    return X_features


# In[23]:


X_train_wavelet = extract_wavelet_features(X_train, wavelet='db4', level=4)
X_val_wavelet =  extract_wavelet_features(X_val, wavelet='db4', level=4)
X_test_wavelet = extract_wavelet_features(X_test, wavelet='db4', level=4)

print(f"X_train_wavelet shape: {X_train_wavelet.shape}")
print(f"X_val_wavelet shape: {X_val_wavelet.shape}")
print(f"X_test_wavelet shape: {X_test_wavelet.shape}")


# In[24]:


output_ranges = np.max(Y_train, axis=0) - np.min(Y_train, axis=0)

# Calculate the inverse of the range to use as scaling factors
scaling_factors = 1 / output_ranges

# Normalize scaling factors to sum up to 1
scaling_factors /= scaling_factors.sum()

print(scaling_factors) 


# # Build Model

# In[25]:


EPOCHS = 200
BATCH_SIZE = 64
PATIENCE = 20


# In[26]:


cleaned_chunks_df[test_mask]['subject_id'].to_numpy()


# In[ ]:





# In[ ]:





# In[27]:


#create model
def create_CNN_model_5_output(shape, start_neuron = 64, kernel_size=9, strides_size=1, max_pool_size=3, dropout=0.3, padding='valid', delta=1.0):    

    input_layer = tf.keras.Input((shape[1], shape[2])) 

    
    conv_1 = tf.keras.layers.Conv1D(start_neuron * 1, kernel_size, strides=strides_size,  padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    conv_1 = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_1)
    conv_1 = tf.keras.layers.Dropout(dropout)(conv_1)
    
    conv_2 = tf.keras.layers.Conv1D(start_neuron * 2, kernel_size, strides=strides_size, padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv_1)
    conv_2 = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_2)
    conv_2 = tf.keras.layers.Dropout(dropout)(conv_2)    
    
    conv_3 = tf.keras.layers.Conv1D(start_neuron * 4, kernel_size, strides=strides_size, padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv_2)
    conv_3 = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_3)
    conv_3 = tf.keras.layers.Dropout(dropout)(conv_3)
    
    
    flattern_layer = tf.keras.layers.Flatten()(conv_3)
    flattern_layer = tf.keras.layers.Dense(start_neuron * 2,activation=tf.keras.layers.LeakyReLU())(flattern_layer)
    
#     output_0 = tf.keras.layers.Dense(32,activation='relu',)(flattern_layer)
#     output_0 = tf.keras.layers.Dropout(dropout)(output_0)    
    output_0 = tf.keras.layers.Dense(1,activation='linear',name='mean_rr')(flattern_layer)
    
    
#     output_1 = tf.keras.layers.Dense(32,activation='relu',)(flattern_layer)
#     output_1 = tf.keras.layers.Dropout(dropout)(output_1)    
    output_1 = tf.keras.layers.Dense(1,activation='linear',name='area_t_to_p')(flattern_layer)
    
#     output_2 = tf.keras.layers.Dense(32,activation='relu',)(flattern_layer)
#     output_2 = tf.keras.layers.Dropout(dropout)(output_2)    
    output_2 = tf.keras.layers.Dense(1,activation='linear',name='area_p_to_t')(flattern_layer)
    
#     output_3 = tf.keras.layers.Dense(32,activation='relu',)(flattern_layer)
#     output_3 = tf.keras.layers.Dropout(dropout)(output_3)   
    output_3 = tf.keras.layers.Dense(1,activation='linear',name='time_t_to_p')(flattern_layer)
    
#     output_4 = tf.keras.layers.Dense(32,activation='relu',)(flattern_layer)    
#     output_4 = tf.keras.layers.Dropout(dropout)(output_4)   
    output_4 = tf.keras.layers.Dense(1,activation='linear',name='time_p_to_t')(flattern_layer)

    
    model = tf.keras.Model(input_layer, [output_0,output_1,output_2,output_3,output_4])
    loss_weights = [scaling_factors[0]*5,scaling_factors[1],scaling_factors[2],scaling_factors[3],scaling_factors[4]]   
    model.compile(optimizer = 'adam',loss = [tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta)],loss_weights=loss_weights)
    return model


# In[28]:


#create model
def create_CNN_LSTM_model_5_output(shape, start_neuron = 64, kernel_size=9, strides_size=1, max_pool_size=3, dropout=0.3, padding='valid', delta=1.0):    
    input_layer = tf.keras.Input((shape[1], shape[2])) 
    
    conv_1 = tf.keras.layers.Conv1D(start_neuron * 1, kernel_size, strides=strides_size,  padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    conv_1 = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_1)
    conv_1 = tf.keras.layers.Dropout(dropout)(conv_1)
    
    conv_2 = tf.keras.layers.Conv1D(start_neuron * 2, kernel_size, strides=strides_size, padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv_1)
    conv_2 = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_2)
    conv_2 = tf.keras.layers.Dropout(dropout)(conv_2)    
    
    lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron, return_sequences=True))(conv_2)
    lstm_1 = tf.keras.layers.Dropout(dropout)(lstm_1)

    lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron * 2, return_sequences=True))(lstm_1)
    lstm_2 = tf.keras.layers.Dropout(dropout)(lstm_2)
    lstm_2 = tf.keras.layers.Flatten()(lstm_2)
    
    output_layer = tf.keras.layers.Dense(start_neuron*2,activation=tf.keras.layers.LeakyReLU())(lstm_2)
    output_layer = tf.keras.layers.Dropout(dropout)(output_layer)
    
    #output_0 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_0 = tf.keras.layers.Dense(1,activation='linear',name='mean_rr')(output_layer)

    #output_1 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_1 = tf.keras.layers.Dense(1,activation='linear',name='area_t_to_p')(output_layer)

    #output_2 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_2 = tf.keras.layers.Dense(1,activation='linear',name='area_p_to_t')(output_layer)

    #output_3 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_3 = tf.keras.layers.Dense(1,activation='linear',name='time_t_to_p')(output_layer)

    #output_4 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_4 = tf.keras.layers.Dense(1,activation='linear',name='time_p_to_t')(output_layer)

    
    model = tf.keras.Model(input_layer, [output_0,output_1,output_2,output_3,output_4])
    loss_weights = [scaling_factors[0]*5,scaling_factors[1],scaling_factors[2],scaling_factors[3],scaling_factors[4]]   
    #loss_weights = [scaling_factors[0]*5,0,0,0,0]  
    model.compile(optimizer = 'adam',loss = [tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta)],loss_weights=loss_weights)
    return model


# In[29]:


#create model
def create_time_wavelet_model_5_output(shape, start_neuron = 64, kernel_size=9, strides_size=1, max_pool_size=3, dropout=0.3, padding='valid', delta=1.0):    
    
    input_layer = tf.keras.Input((shape[1], shape[2]), name='input_time') 
    input_layer_wavelet = tf.keras.Input((shape[4], shape[2]), name='input_wavelet') 

    #time domain
    conv_1 = tf.keras.layers.Conv1D(start_neuron * 1, kernel_size, strides=strides_size,  padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    conv_1 = tf.keras.layers.Conv1D(start_neuron * 1, kernel_size, strides=strides_size,  padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv_1)
    conv_1 = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_1)
    conv_1 = tf.keras.layers.Dropout(dropout)(conv_1)
    
    conv_2 = tf.keras.layers.Conv1D(start_neuron * 2, kernel_size, strides=strides_size, padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv_1)
    conv_2 = tf.keras.layers.Conv1D(start_neuron * 2, kernel_size, strides=strides_size, padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv_2)
    conv_2 = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_2)
    conv_2 = tf.keras.layers.Dropout(dropout)(conv_2)    
    
    lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron, return_sequences=True))(conv_2)
    lstm_1 = tf.keras.layers.Dropout(dropout)(lstm_1)

    lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron * 2, return_sequences=True))(lstm_1)
    lstm_2 = tf.keras.layers.Dropout(dropout)(lstm_2)
    lstm_2 = tf.keras.layers.Flatten()(lstm_2)

    #wavelet domain
    conv_1_wavelet = tf.keras.layers.Conv1D(start_neuron * 1, kernel_size, strides=strides_size,  padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer_wavelet)
    conv_1_wavelet = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_1_wavelet)
    conv_1_wavelet = tf.keras.layers.Dropout(dropout)(conv_1_wavelet)
    
    conv_2_wavelet = tf.keras.layers.Conv1D(start_neuron * 2, kernel_size, strides=strides_size, padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv_1_wavelet)
    conv_2_wavelet = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(conv_2_wavelet)
    conv_2_wavelet = tf.keras.layers.Dropout(dropout)(conv_2_wavelet)    
    
    lstm_1_wavelet = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron, return_sequences=True))(conv_2_wavelet)
    lstm_1_wavelet = tf.keras.layers.Dropout(dropout)(lstm_1_wavelet)

    lstm_2_wavelet = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron * 2, return_sequences=True))(lstm_1_wavelet)
    lstm_2_wavelet = tf.keras.layers.Dropout(dropout)(lstm_2_wavelet)
    lstm_2_wavelet = tf.keras.layers.Flatten()(lstm_2_wavelet)    

    concat_layer = tf.keras.layers.Concatenate()([lstm_2, lstm_2_wavelet])
    
    output_layer = tf.keras.layers.Dense(start_neuron*2,activation=tf.keras.layers.LeakyReLU())(concat_layer)
    output_layer = tf.keras.layers.Dropout(dropout)(output_layer)
    
    #output_0 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_0 = tf.keras.layers.Dense(1,activation='linear',name='mean_rr')(output_layer)

    #output_1 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_1 = tf.keras.layers.Dense(1,activation='linear',name='area_t_to_p')(output_layer)

    #output_2 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_2 = tf.keras.layers.Dense(1,activation='linear',name='area_p_to_t')(output_layer)

    #output_3 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_3 = tf.keras.layers.Dense(1,activation='linear',name='time_t_to_p')(output_layer)

    #output_4 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_4 = tf.keras.layers.Dense(1,activation='linear',name='time_p_to_t')(output_layer)

    
    model = tf.keras.Model(inputs=[input_layer, input_layer_wavelet], outputs=[output_0,output_1,output_2,output_3,output_4])
    loss_weights = [scaling_factors[0]*5,scaling_factors[1],scaling_factors[2],scaling_factors[3],scaling_factors[4]]   
    #loss_weights = [scaling_factors[0]*5,0,0,0,0]  
    model.compile(optimizer = 'adam',loss = [tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta)],loss_weights=loss_weights)
    #model.compile(optimizer = 'adam',loss = ['mae','mae','mae','mae','mae'],loss_weights=loss_weights)
    return model


# In[30]:


#create model
def custom_cnn_layers(input_layer, start_neuron, kernel_size, strides_size, max_pool_size, dropout, padding='valid', delta=1.0):
    layer = tf.keras.layers.Conv1D(start_neuron, kernel_size, strides=strides_size,  padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    layer = tf.keras.layers.Conv1D(start_neuron, kernel_size, strides=strides_size,  padding=padding, activation=tf.keras.layers.LeakyReLU(),kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
    layer = tf.keras.layers.MaxPool1D(max_pool_size,  padding=padding)(layer)
    layer = tf.keras.layers.Dropout(dropout)(layer)
    return layer

    
def create_3_input_model_5_output(shape, start_neuron = 64, kernel_size=9, strides_size=1, max_pool_size=3, dropout=0.3, padding='valid', delta=1.0):
    input_layer = tf.keras.Input((shape[1], shape[2]), name='input_time') 
    input_layer_wavelet = tf.keras.Input((shape[4], shape[2]), name='input_wavelet') 
    input_layer_feature = tf.keras.Input((shape[6],),name='input_feature')
    
    #time domain
    conv_1 = custom_cnn_layers(input_layer, start_neuron = start_neuron, kernel_size = kernel_size, strides_size = strides_size, max_pool_size = 5, dropout = dropout, padding='same', delta=1.0)
    conv_2 = custom_cnn_layers(conv_1, start_neuron = start_neuron * 2, kernel_size = kernel_size, strides_size = strides_size, max_pool_size = 5, dropout = dropout, padding='same', delta=1.0)
    
    conv_3 = custom_cnn_layers(conv_2, start_neuron = start_neuron * 4, kernel_size = kernel_size, strides_size = strides_size, max_pool_size = 3, dropout = dropout, padding='same', delta=1.0)
    conv_4 = custom_cnn_layers(conv_3, start_neuron = start_neuron * 8, kernel_size = kernel_size, strides_size = strides_size, max_pool_size = 2, dropout = 0, padding='same', delta=1.0)
    conv_4 = tf.keras.layers.GlobalAveragePooling1D()(conv_4)
    conv_4 = tf.keras.layers.Dropout(dropout)(conv_4)

    lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron * 2, return_sequences=True))(conv_2)
    #lstm_1 = tf.keras.layers.MaxPool1D(3,  padding=padding)(lstm_1)
    lstm_1 = tf.keras.layers.Dropout(dropout)(lstm_1)
    
    lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron * 4, return_sequences=True))(lstm_1)
    lstm_2 = tf.keras.layers.GlobalAveragePooling1D()(lstm_2)
    lstm_2 = tf.keras.layers.Dropout(dropout)(lstm_2)

    #wavelet domain
    conv_1_wavelet = custom_cnn_layers(input_layer_wavelet, start_neuron = start_neuron, kernel_size = kernel_size, strides_size = strides_size, max_pool_size = 5, dropout = dropout, padding='same', delta=1.0)
    conv_2_wavelet = custom_cnn_layers(conv_1_wavelet, start_neuron = start_neuron * 2, kernel_size = kernel_size, strides_size = strides_size, max_pool_size = 5, dropout = dropout, padding='same', delta=1.0)
    
    conv_3_wavelet = custom_cnn_layers(conv_2_wavelet, start_neuron = start_neuron * 4, kernel_size = kernel_size, strides_size = strides_size, max_pool_size = 3, dropout = dropout, padding='same', delta=1.0)
    conv_4_wavelet = custom_cnn_layers(conv_3_wavelet, start_neuron = start_neuron * 8, kernel_size = kernel_size, strides_size = strides_size, max_pool_size = 2, dropout = 0, padding='same', delta=1.0)
    conv_4_wavelet = tf.keras.layers.GlobalAveragePooling1D()(conv_4_wavelet)
    conv_4_wavelet = tf.keras.layers.Dropout(dropout)(conv_4_wavelet)

    lstm_1_wavelet = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron * 2, return_sequences=True))(conv_2_wavelet)
    #lstm_1_wavelet = tf.keras.layers.MaxPool1D(3,  padding=padding)(lstm_1_wavelet)
    lstm_1_wavelet = tf.keras.layers.Dropout(dropout)(lstm_1_wavelet)
    
    lstm_2_wavelet = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(start_neuron * 4, return_sequences=True))(lstm_1_wavelet)
    lstm_2_wavelet = tf.keras.layers.GlobalAveragePooling1D()(lstm_2_wavelet)
    lstm_2_wavelet = tf.keras.layers.Dropout(dropout)(lstm_2_wavelet)

    
    concat_layer = tf.keras.layers.Concatenate()([conv_4,lstm_2,conv_4_wavelet,lstm_2_wavelet,input_layer_feature])
    output_layer = tf.keras.layers.Dense(start_neuron*8,activation=tf.keras.layers.LeakyReLU())(concat_layer)
    output_layer = tf.keras.layers.Dropout(dropout)(output_layer)
    
    #output_0 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_0 = tf.keras.layers.Dense(1,activation='linear',name='mean_rr')(output_layer)

    #output_1 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_1 = tf.keras.layers.Dense(1,activation='linear',name='area_t_to_p')(output_layer)

    #output_2 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_2 = tf.keras.layers.Dense(1,activation='linear',name='area_p_to_t')(output_layer)

    #output_3 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_3 = tf.keras.layers.Dense(1,activation='linear',name='time_t_to_p')(output_layer)

    #output_4 = tf.keras.layers.Dense(start_neuron*2,activation='relu')(output_layer)
    output_4 = tf.keras.layers.Dense(1,activation='linear',name='time_p_to_t')(output_layer)

    
    model = tf.keras.Model(inputs=[input_layer, input_layer_wavelet, input_layer_feature], outputs=[output_0,output_1,output_2,output_3,output_4])
    loss_weights = [scaling_factors[0]*5,scaling_factors[1],scaling_factors[2],scaling_factors[3],scaling_factors[4]]   
    #loss_weights = [scaling_factors[0]*5,0,0,0,0]  
    model.compile(optimizer = 'adam',loss = [tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta),tf.keras.losses.Huber(delta=delta)],loss_weights=loss_weights)
    #model.compile(optimizer = 'adam',loss = ['mae','mae','mae','mae','mae'],loss_weights=loss_weights)
    return model


# In[31]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt

def piecewise_regression(x, y, breakpoints, ax):
    segments = np.split(np.argsort(x), np.searchsorted(x[np.argsort(x)], breakpoints))
    segment_texts = []
    
    for i,segment in enumerate(segments):
        if len(segment) > 1:  # Fit line if segment contains more than one point
            slope, intercept, _, _, _ = linregress(x[segment], y[segment])
            ax.plot(x[segment], slope * x[segment] + intercept, color='red', linewidth=2)
            
            # Calculate Pearson correlation coefficient and p-value for each segment
            corr, p_val = pearsonr(x[segment], y[segment])
            spearman_corr, spearman_p_val = spearmanr(x[segment], y[segment])
            segment_texts.append(f'Segment {i}: Pearsonr: {corr:.2f},  Spearman: {spearman_corr:.2f}')
    
    return segment_texts
    
if(not(os.path.exists(OUTPUT_DIR))):
    os.makedirs(OUTPUT_DIR)
    
# Define the grid search parameters
param_grid = {
    'start_neuron': [32],
    'kernel_size': [5,7,9,11,13],
    'strides_size': [1,2,3,4],
    'max_pool_size': [5],
    'dropout': [0.3,0.5],
    'padding' :['same'],
    'batch_size' : [64],
    'huber_delta' : [0.75],
}

# Prepare callbacks


# Number of epochs
EPOCHS = 200
PATIENCE = 20

# Initialize a list to store the loss history for all configurations
loss_history = []

# Perform grid search
for start_neuron in param_grid['start_neuron']:
    for kernel_size in param_grid['kernel_size']:
        for strides_size in param_grid['strides_size']:
            for max_pool_size in param_grid['max_pool_size']:
                for dropout in param_grid['dropout']:
                    for padding in param_grid['padding']:
                        for batch_size in param_grid['batch_size']:
                            for delta in param_grid['huber_delta']:
                                print(f'start_neuron={start_neuron}, kernel_size={kernel_size}, strides_size={strides_size}, max_pool_size={max_pool_size}, dropout={dropout}, padding={padding}, batch_size={batch_size}, huber_delta={delta}')
                                
                                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=PATIENCE//2,verbose=1, min_delta=0.00001,)
                                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights = True, mode = 'min')
                                # Create the model
                                model = create_3_input_model_5_output((X_train.shape + X_train_wavelet.shape + X_train_feature.shape), start_neuron=start_neuron, kernel_size=kernel_size,
                                                                  strides_size=strides_size, max_pool_size=max_pool_size, 
                                                                  dropout=dropout, padding=padding, delta=delta)
                                model.summary()
    
    
                                # Train the model
                                history = model.fit([X_train,X_train_wavelet,X_train_feature], [Y_train[:,0],Y_train[:,1],Y_train[:,2],Y_train[:,3],Y_train[:,4]], 
                                                    validation_data=([X_val,X_val_wavelet,X_val_feature], [Y_val[:,0],Y_val[:,1],Y_val[:,2],Y_val[:,3],Y_val[:,4]]), 
                                                    epochs=EPOCHS, batch_size=BATCH_SIZE,verbose=0, shuffle=True, 
                                                    callbacks=[early_stopping, reduce_lr])



                                min_train_loss = min(history.history['loss'])
                                min_val_lost = min(history.history['val_loss'])
                                # Save loss history for the current configuration
                                config_loss_history = {
                                    'config': f'start_neuron={start_neuron}, kernel_size={kernel_size}, strides_size={strides_size}, max_pool_size={max_pool_size}, dropout={dropout}, padding={padding}, batch_size={batch_size}, huber_delta={delta}',
                                    'training_loss': history.history['loss'],
                                    'validation_loss': history.history['val_loss'],
                                    'min' : f'Best_train_loss{min_train_loss}, best_val_loss{min_val_lost}'
                                }
                                loss_history.append(config_loss_history)
    
                                # Evaluate the model
                                Y_pred = model.predict([X_test,X_test_wavelet,X_test_feature],verbose=0)

                                plt.clf()
                                unique_subject_ids = np.unique(test_subject_id)
                                colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_subject_ids)))
                                color_map = {subject_id: colors[i] for i, subject_id in enumerate(unique_subject_ids)}


                                # Initialize the figure and subplots
                                # Create plots
                                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                                axes = axes.flatten()
                                output_names = ['Mean Respiration Rate', 'Area T to P', 'Area P to T', 'Time T to P', 'Time P to T']
    
                                for i in range(5):
                                    # Calculate the Pearson correlation coefficient
                                    correlation, p_value = pearsonr(Y_test[:, i], Y_pred[i].flatten())
                                    print(f"Output {i}: Pearson Correlation: {correlation}")
                                    print(f"Output {i}: P-value: {p_value}")
                                
                                    # # Create scatter plot
                                    # axes[i].scatter(Y_test[:, i], Y_pred[i].flatten(), alpha=0.6, edgecolors='w', linewidth=0.5, label='Data points')
                                    
                                    for subject_id in unique_subject_ids:
                                        mask = test_subject_id == subject_id
                                        axes[i].scatter(Y_test[mask, i], Y_pred[i].flatten()[mask], alpha=0.6, edgecolors='w', linewidth=0.5, label=f'Subject {subject_id}', color=color_map[subject_id])
                                
                                    # Define breakpoints
                                    breakpoints = np.percentile(Y_test[:, i], [33, 66])
                                
                                    # Fit and plot piecewise regression
                                    segment_texts = piecewise_regression(Y_test[:, i], Y_pred[i].flatten(), breakpoints, axes[i])
                                
                                    axes[i].set_title(f'Actual vs Predicted {output_names[i]}')
                                    axes[i].set_xlabel('Actual')
                                    axes[i].set_ylabel('Predicted')
                                    axes[i].grid(True)
                                    axes[i].legend()
                                
                                    # Add text box with correlation and p-value
                                    textstr = f'Pearson Correlation: {correlation:.2f}\n' + '\n'.join(segment_texts)
                                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                                    axes[i].text(0.05, 0.95, textstr, transform=axes[i].transAxes, fontsize=10,
                                                 verticalalignment='top', bbox=props)
                                
                                # Plot the model loss
                                axes[5].plot(history.history['loss'], label='Training Loss')
                                axes[5].plot(history.history['val_loss'], label='Validation Loss')
                                axes[5].set_title('Model Loss')
                                axes[5].set_ylabel('Loss')
                                axes[5].set_xlabel('Epoch')
                                axes[5].legend(loc='upper right')

                                # Adjust layout to prevent overlap
                                plt.tight_layout()
                                
                                # Ensure x and y axes have the same scale
                                for i, ax in enumerate(axes):
                                    if i < len(output_names):
                                        ax.set_xlim([min(Y_test[:,i].min(), Y_pred[i].min()), max(Y_test[:,i].max(), Y_pred[i].max())])
                                        ax.set_ylim([min(Y_test[:,i].min(), Y_pred[i].min()), max(Y_test[:,i].max(), Y_pred[i].max())])
    
                                # Save the plot
                                plt.savefig(OUTPUT_DIR + f'grid_search_{start_neuron}_{kernel_size}_{strides_size}_{max_pool_size}_{dropout}_{padding}_{batch_size}_{delta}.png')
                                plt.close()

# Save loss history to a text file
with open(OUTPUT_DIR + 'loss_history.txt', 'w') as f:
    for entry in loss_history:
        f.write(f"Configuration: {entry['config']}\n")
        f.write(f"Training Loss: {entry['training_loss']}\n")
        f.write(f"Validation Loss: {entry['validation_loss']}\n")
        f.write(f"{entry['min']}\n")
        f.write("\n")

print("Grid search completed and results saved.")


# In[ ]:





# In[41]:


# param_grid = {
#     'start_neuron': [32,64],
#     'kernel_size': [5,7,9,11],
#     'strides_size': [1,2,3,4],
#     'max_pool_size': [2,3],
#     'dropout': [0.3,0.4,0.5],
#     'padding' :['same','valid'],
#     'batch_size' : [64],
#     'huber_delta' : [0.5,1],
    
# }


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




