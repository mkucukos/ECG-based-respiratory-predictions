#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[13]:


import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.signal import resample
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, linregress
import random
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.colors as mcolors
from scipy.stats import norm


# In[ ]:





# In[3]:


RUN = 'preprocessed_data_EDA/'

CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/../dataset/'
MASTER_OUTPUT_DIR = CURRENT_DIR + '/../output/'
OUTPUT_DIR = MASTER_OUTPUT_DIR + RUN
print(DATA_DIR)
print(OUTPUT_DIR)


# In[4]:


os.makedirs(OUTPUT_DIR,exist_ok=True)


# In[ ]:





# In[ ]:





# # Load Data

# In[5]:


data_df = pd.read_pickle(DATA_DIR + 'fantasia_dataset_preprocessing.plk')
data_df


# In[6]:


sorted_data = data_df[data_df['Mean_Respiration_Rate'] < 15]
sorted_data = sorted_data.sort_values(by=['Mean_Respiration_Rate'])
sorted_data


# In[7]:


mean_respiration_rate = data_df['Mean_Respiration_Rate'].dropna()  # Drop NaN values for the histogram

# Plot the histogram
count, bins, ignored = plt.hist(mean_respiration_rate, bins=30, density=False, alpha=0.6, color='g')

# Fit a normal distribution to the data
mu, std = norm.fit(mean_respiration_rate)

# Plot the Gaussian distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

# Add vertical lines for -/+ 2 std
plt.axvline(mu - 2*std, color='r', linestyle='dashed', linewidth=2)
plt.axvline(mu + 2*std, color='r', linestyle='dashed', linewidth=2)

title = f"Mean Respiration Rate results: mu = {mu:.2f},  std = {std:.2f}"
plt.title(title)

plt.xlabel('Mean Respiration Rate')
plt.ylabel('count')
plt.savefig(OUTPUT_DIR + 'respiration_distribution.png')
plt.show()


# In[12]:


# Plot the histogram
plt.figure(figsize=(18,6))
count, bins, ignored = plt.hist(data_df['snr'], bins=300, density=True, alpha=0.6, color='g')

# Fit a normal distribution to the data
mu, std = norm.fit(sorted_data['snr'])

# Plot the Gaussian distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.axvline(mu , color='blue', linestyle='dashed', linewidth=2)
plt.axvline(mu - 2*std, color='r', linestyle='dashed', linewidth=2)
plt.axvline(mu + 2*std, color='r', linestyle='dashed', linewidth=2)

title = f"Fit results: mu = {mu:.2f},  std = {std:.2f}"
plt.title(title)

plt.xlabel('SNR Value')
plt.ylabel('Probability Density')
plt.savefig(OUTPUT_DIR + 'snr_distribution.png')
plt.show()


# In[16]:


import seaborn as sns
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Define the columns to plot against Breath_rate_variability
columns_to_plot = ['Total_peaks', 'Total_trough', 'hrv', 'Mean_Respiration_Rate','snr']

# Iterate over the columns and create a scatter plot with a regression line
for ax, col in zip(axes.flatten(), columns_to_plot):
    sns.scatterplot(x='Breath_rate_variability', y=col, data=sorted_data, ax=ax)
    ax.set_title(f'Breath_rate_variability vs {col}')
    ax.set_xlabel('Breath_rate_variability')
    ax.set_ylabel(col)
    ax.grid()
    if col == 'hrv':
        ax.set_ylim((0,1000))
    
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'breathing_rate_variability_comparision.png')
plt.show()


# In[8]:


# Sample DataFrame
# df = pd.DataFrame({
#     'resp': [list_of_int1, list_of_int2, ..., list_of_intN],
#     'ecg': [list_of_int1, list_of_int2, ..., list_of_intN],
#     'Mean_Respiration_Rate': [rate1, rate2, ..., rateN]
# })  # Replace with your actual DataFrame

# Assuming sorted_data is already defined and contains the columns 'resp', 'ecg', and 'Mean_Respiration_Rate'
sorted_data = sorted_data[['resp', 'ecg', 'Mean_Respiration_Rate']]

# Randomly select 10 indices
random_indices = random.sample(range(len(sorted_data)), 10)

# Plot
fig, axes = plt.subplots(10, 2, figsize=(15, 30))  # Create a 10x2 grid of subplots

for i, idx in enumerate(random_indices):
    resp = sorted_data.iloc[idx]['resp']
    ecg = sorted_data.iloc[idx]['ecg']
    mean_resp_rate = sorted_data.iloc[idx]['Mean_Respiration_Rate']
    
    # Plot 'resp'
    axes[i, 0].plot(ecg, color='b')
    axes[i, 0].set_title(f'ECG: Mean Respiration Rate = {mean_resp_rate}')
    
    # Plot 'ecg'
    axes[i, 1].plot(resp, color='g')
    axes[i, 1].set_title(f'Respiration: Mean Respiration Rate = {mean_resp_rate}')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'ecg_respiration_comparision.png')
plt.show()


# In[9]:


# Assuming sorted_data is already defined and contains the columns 'resp', 'ecg', and 'Mean_Respiration_Rate'
filter_df = data_df.query("Breath_rate_variability > 1000 and Breath_rate_variability < 1500").dropna()


filter_df = filter_df.sort_values(by=['Breath_rate_variability'])
filter_df = filter_df[['resp', 'ecg', 'Mean_Respiration_Rate','Breath_rate_variability']]


# Randomly select 10 indices
random_indices = random.sample(range(len(filter_df)), 10)

# Plot
fig, axes = plt.subplots(5, 2, figsize=(15, 15))  # Create a 10x2 grid of subplots
axes = axes.flatten()

for i, idx in enumerate(random_indices):
    resp = filter_df.iloc[idx]['resp']
    ecg = filter_df.iloc[idx]['ecg']
    mean_resp_rate = filter_df.iloc[idx]['Mean_Respiration_Rate']
    brv = filter_df.iloc[idx]['Breath_rate_variability']
    
    # Plot 'resp'
    axes[i].plot(resp, color='b')
    axes[i].set_title(f'Mean Respiration Rate: {mean_resp_rate}\nBreath rate variability:{brv}')
    
fig.suptitle("Breath rate variability between 1300 - 1500 and Respiration Rate > 12\n", fontsize = 20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'respiration_quality_check.png')
plt.show()


# In[10]:


filter_df


# In[ ]:





# In[ ]:





# In[ ]:




