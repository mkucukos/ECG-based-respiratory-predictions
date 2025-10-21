#!/usr/bin/env python
# coding: utf-8

# In[2]:



# In[3]:


import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.signal import resample
from scipy import stats


# In[2]:

RUN = 'initial_data_EDA/'
CURRENT_DIR = os.getcwd() + '/..'
DATA_DIR = CURRENT_DIR + '/dataset/'
MASTER_OUTPUT_DIR = CURRENT_DIR + '/output/'
OUTPUT_DIR = MASTER_OUTPUT_DIR + RUN
print(DATA_DIR)
print(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR,exist_ok=True)

# In[3]:


# Load the dataset
file_path = DATA_DIR + 'fantasia_dataset.plk'
df = pd.read_pickle(file_path)
df


# In[4]:


# Need to resample 'f2y02' from 333 Hz to 250 Hz
subject_id = 'f2y02'
original_sample_rate = df[df['subject_id'] == subject_id]['sample_rate'].iloc[0]
target_sample_rate = 250

downsample_factor = target_sample_rate / original_sample_rate
n_samples = int(len(df[df['subject_id'] == subject_id]) * downsample_factor)

# Filter the DataFrame
df_subject = df[df['subject_id'] == subject_id].copy()
df_new = pd.DataFrame(columns=df_subject.columns.to_list())

# Downsample the 'ecg' and 'resp' columns

df_new['ecg'] = resample(df_subject['ecg'], n_samples)
df_new['resp'] = resample(df_subject['resp'], n_samples)
df_new['subject_id'] = df_subject['subject_id'].iloc[0]
df_new['sample_rate'] = target_sample_rate
df_new['sample'] = df_new.index.to_list()

#replace value
df = df[df['subject_id'] != subject_id]
df = pd.concat([df,df_new],ignore_index=True).reset_index(drop=True)
# # Verify the update
# print(df[df['subject_id'] == subject_id])


# In[5]:


df


# In[ ]:





# In[6]:


# Group by subject_id and choose one random subject
grouped = df.groupby('subject_id')
random_subject = np.random.choice(df['subject_id'].unique())

# Filter the dataframe for the chosen subject
#subject_df = grouped.get_group(random_subject)
subject_df = grouped.get_group('f2y02')

# Convert samples to seconds
sample_rate = 250  # 250 samples per second
time_in_seconds = subject_df['sample'].iloc[1:7500] / sample_rate

# Plot the data
plt.figure(figsize=(14, 12))

# Plot ECG data
plt.subplot(2, 1, 1)
plt.plot(time_in_seconds, subject_df['ecg'].iloc[1:7500], label='ECG', color='blue')
plt.title(f'ECG and Respiration for Subject {random_subject}')
plt.xlabel('Time (seconds)')
plt.ylabel('ECG Value')
plt.legend()
plt.grid(True)

# Plot respiration data
plt.subplot(2, 1, 2)
plt.plot(time_in_seconds, subject_df['resp'].iloc[1:7500], label='Respiration', color='green')
plt.xlabel('Time (seconds)')
plt.ylabel('Respiration Value')
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'example_data.png')
plt.show()


# In[7]:


# Ensure the indices for slicing are within the bounds of the DataFrame
sample_rate = 250  # 250 samples per second
start_index = 1
end_index = min(7500, len(subject_df))

# Convert samples to seconds
time_in_seconds = subject_df['sample'].iloc[start_index:end_index] / sample_rate

# Respiration data for the selected range
respiration = subject_df['resp'].iloc[start_index:end_index]

# Clean and process the respiration data using nk.rsp_process
rsp_signals, rsp_info = nk.rsp_process(subject_df['resp'], sampling_rate=sample_rate, method='khodadad2018')

# Extract the cleaned respiration signal, peaks, troughs, and respiration rate
cleaned_respiration = rsp_signals["RSP_Clean"].iloc[start_index:end_index]
peaks = rsp_signals["RSP_Peaks"].iloc[start_index:end_index].values
troughs = rsp_signals["RSP_Troughs"].iloc[start_index:end_index].values
respiration_rate = rsp_signals["RSP_Rate"].iloc[start_index:end_index].values

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 18))

# Plot raw respiration data
axs[0].plot(time_in_seconds, respiration, label='Raw Respiration', color='green', alpha=0.6)
axs[0].set_title('Raw Respiration Data')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Respiration Value')
axs[0].legend()
axs[0].grid(True)

# Plot cleaned respiration data with peaks and troughs
axs[1].plot(time_in_seconds, cleaned_respiration, label='Cleaned Respiration (NeuroKit)', color='black')
axs[1].scatter(time_in_seconds[peaks == 1], cleaned_respiration[peaks == 1], color='red', marker='o', s=150, label='Peaks')
axs[1].scatter(time_in_seconds[troughs == 1], cleaned_respiration[troughs == 1], color='blue', marker='o', s=150, label='Troughs')
axs[1].set_title('Cleaned Respiration Data with Peaks and Troughs (Neurokit)')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Respiration Value')
axs[1].legend()
axs[1].grid(True)

# Plot respiration rate
axs[2].plot(time_in_seconds, respiration_rate, label='Respiration Rate (Breaths per Minute)', color='purple', linestyle='--')
axs[2].set_title('Respiration Rate')
axs[2].set_xlabel('Time (seconds)')
axs[2].set_ylabel('Breaths per Minute')
axs[2].legend()
axs[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'example_peaks_trough_neurokit_filters.png')
plt.show()


# In[8]:


rsp_signals


# In[9]:


subject_df


# In[10]:


rsp_info


# In[11]:


# Ensure the indices for slicing are within the bounds of the DataFrame
sample_rate = 250  # 250 samples per second
start_index = 1
end_index = min(7500, len(subject_df))

# Convert samples to seconds
time_in_seconds = subject_df['sample'][start_index:end_index] / sample_rate

# Respiration data for the selected range
respiration = subject_df['resp'][start_index:end_index]

# Apply a low-pass Butterworth filter to the respiration data
def low_pass_filter(data, cutoff_freq, sample_rate):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

cutoff_frequency = 1  # Set the cutoff frequency to 1 Hz
filtered_respiration = low_pass_filter(subject_df['resp'], cutoff_frequency, sample_rate)

# Clean and process the filtered respiration data using nk.rsp_process
rsp_signals, rsp_info = nk.rsp_process(filtered_respiration, sampling_rate=sample_rate, method='khodadad2018')

# Extract the cleaned respiration signal, peaks, troughs, and respiration rate
cleaned_respiration = rsp_signals["RSP_Clean"].iloc[start_index:end_index]
peaks = rsp_signals["RSP_Peaks"].iloc[start_index:end_index].values
troughs = rsp_signals["RSP_Troughs"].iloc[start_index:end_index].values
respiration_rate = rsp_signals["RSP_Rate"].iloc[start_index:end_index].values

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 18))

# Plot raw respiration data
axs[0].plot(time_in_seconds, respiration, label='Raw Respiration', color='green', alpha=0.6)
axs[0].set_title('Raw Respiration Data')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Respiration Value')
axs[0].legend()
axs[0].grid(True)

# Plot cleaned respiration data with peaks and troughs
axs[1].plot(time_in_seconds, cleaned_respiration, label='Cleaned Respiration (Butterworth + Neurokit)', color='black')
axs[1].scatter(time_in_seconds[peaks == 1], cleaned_respiration[peaks == 1], color='red', marker='o', s=150, label='Peaks')
axs[1].scatter(time_in_seconds[troughs == 1], cleaned_respiration[troughs == 1], color='blue', marker='o', s=150, label='Troughs')
axs[1].set_title('Cleaned Respiration Data with Peaks and Troughs (Butterworth + Neurokit)')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Respiration Value')
axs[1].legend()
axs[1].grid(True)

# Plot respiration rate
axs[2].plot(time_in_seconds, respiration_rate, label='Respiration Rate (Breaths per Minute)', color='purple', linestyle='--')
axs[2].set_title('Respiration Rate')
axs[2].set_xlabel('Time (seconds)')
axs[2].set_ylabel('Breaths per Minute')
axs[2].legend()
axs[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'example_peaks_trough_butterworth_and_neurokit_filters.png')
plt.show()


# In[12]:


# Ensure the indices for slicing are within the bounds of the DataFrame
sample_rate = 250  # 250 samples per second
start_index = 1
end_index = min(7500, len(subject_df))

# Convert samples to seconds
time_in_seconds = subject_df['sample'].iloc[start_index:end_index] / sample_rate

# Respiration data for the selected range
respiration = subject_df['resp'].iloc[start_index:end_index]

# Apply a low-pass Butterworth filter to the respiration data
def low_pass_filter(data, cutoff_freq, sample_rate):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

cutoff_frequency = 1  # Set the cutoff frequency to 1 Hz
filtered_respiration = low_pass_filter(subject_df['resp'], cutoff_frequency, sample_rate)

# Clean and process the filtered respiration data using nk.rsp_process
rsp_signals, rsp_info = nk.rsp_process(filtered_respiration, sampling_rate=sample_rate, method='khodadad2018')

# Extract the cleaned respiration signal, peaks, troughs, and respiration rate
cleaned_respiration = rsp_signals["RSP_Clean"][start_index:end_index]
peaks = rsp_signals["RSP_Peaks"][start_index:end_index].values
troughs = rsp_signals["RSP_Troughs"][start_index:end_index].values
respiration_rate = rsp_signals["RSP_Rate"][start_index:end_index].values

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 18))

# Plot raw respiration data
axs[0].plot(time_in_seconds, respiration, label='Raw Respiration', color='green', alpha=0.6)
axs[0].set_title('Raw Respiration Data')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Respiration Value')
axs[0].legend()
axs[0].grid(True)

# Plot cleaned respiration data with peaks and troughs
axs[1].plot(time_in_seconds, cleaned_respiration, label='Cleaned Respiration (Butterworth + Neurokit)', color='black')
axs[1].scatter(time_in_seconds[peaks == 1], cleaned_respiration[peaks == 1], color='red', marker='o', s=150, label='Peaks')
axs[1].scatter(time_in_seconds[troughs == 1], cleaned_respiration[troughs == 1], color='blue', marker='o', s=150, label='Troughs')

# Calculate and shade tidal volumes
shaded_areas_peak_to_trough = []
shaded_areas_trough_to_peak = []

for i in range(1, len(peaks)):
    if peaks[i] == 1 and i < len(troughs) - 1 and troughs[i] == 0:
        # Shade area from peak to next trough
        next_trough_idx = np.where(troughs[i:] == 1)[0]
        if next_trough_idx.size > 0:
            next_trough_idx = next_trough_idx[0] + i
            axs[1].fill_between(time_in_seconds[i:next_trough_idx], 0, cleaned_respiration[i:next_trough_idx], color='lightblue', alpha=0.5)
            area = np.trapz(cleaned_respiration[i:next_trough_idx], time_in_seconds[i:next_trough_idx])
            shaded_areas_peak_to_trough.append(area)
    elif troughs[i] == 1 and i < len(peaks) - 1 and peaks[i] == 0:
        # Shade area from trough to next peak
        next_peak_idx = np.where(peaks[i:] == 1)[0]
        if next_peak_idx.size > 0:
            next_peak_idx = next_peak_idx[0] + i
            axs[1].fill_between(time_in_seconds[i:next_peak_idx], 0, cleaned_respiration[i:next_peak_idx], color='lightgreen', alpha=0.5)
            area = np.trapz(cleaned_respiration[i:next_peak_idx], time_in_seconds[i:next_peak_idx])
            shaded_areas_trough_to_peak.append(area)

# Sum the absolute values of shaded areas
sum_peak_to_trough = np.sum(np.abs(shaded_areas_peak_to_trough))
sum_trough_to_peak = np.sum(np.abs(shaded_areas_trough_to_peak))

# Display the sums as text in the second subplot
textstr = '\n'.join((
    r'$\sum_{peak \rightarrow trough}=%.2f$' % (sum_peak_to_trough,),
    r'$\sum_{trough \rightarrow peak}=%.2f$' % (sum_trough_to_peak,)
))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs[1].text(0.95, 0.05, textstr, transform=axs[1].transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

axs[1].set_title('Cleaned Respiration Data with Peaks and Troughs (Butterworth + Neurokit)')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Respiration Value')
axs[1].legend()
axs[1].grid(True)

# Calculate and annotate the average respiration rate
average_respiration_rate = np.mean(respiration_rate)
textstr_rate = f'Average Respiration Rate: {average_respiration_rate:.2f} BPM'
props_rate = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs[2].text(0.95, 0.05, textstr_rate, transform=axs[2].transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right', bbox=props_rate)

# Plot respiration rate
axs[2].plot(time_in_seconds, respiration_rate, label='Respiration Rate (Breaths per Minute)', color='purple', linestyle='--')
axs[2].set_title('Respiration Rate')
axs[2].set_xlabel('Time (seconds)')
axs[2].set_ylabel('Breaths per Minute')
axs[2].legend()
axs[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'example_peaks_trough_exhale_inhale_area.png')
plt.show()

# Print the shaded areas
print("Shaded Areas from Peak to Next Trough:", shaded_areas_peak_to_trough)
print("Shaded Areas from Trough to Next Peak:", shaded_areas_trough_to_peak)








