#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


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


# In[5]:


CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/../dataset/'


# In[6]:


# Load the dataset
file_path = DATA_DIR + 'fantasia_dataset.plk'
df = pd.read_pickle(file_path)
df


# In[8]:


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




# In[13]:


def calculate_respiration_metrics(respiration_data, sample_rate=250, cutoff_frequency=1):
    """
    Calculate respiration metrics including mean respiration rate, areas from peak to trough, and trough to peak.
    
    Args:
    - respiration_data (array-like): The respiration signal data.
    - sample_rate (int): The sampling rate of the data.
    - cutoff_frequency (int): The cutoff frequency for the low-pass filter.
    
    Returns:
    - dict: A dictionary containing mean respiration rate, areas from peak to trough, and trough to peak.
    """
    
    # Apply low-pass Butterworth filter
    def low_pass_filter(data, cutoff_freq, sample_rate):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    filtered_respiration = low_pass_filter(respiration_data, cutoff_frequency, sample_rate)

    # Clean and process the filtered respiration data
    rsp_signals, rsp_info = nk.rsp_process(filtered_respiration, sampling_rate=sample_rate, method='khodadad2018')
    cleaned_respiration = rsp_signals["RSP_Clean"]
    peaks = rsp_signals["RSP_Peaks"].values
    troughs = rsp_signals["RSP_Troughs"].values
    respiration_rate = rsp_signals["RSP_Rate"].values

    # Calculate mean respiration rate
    mean_respiration_rate = np.mean(respiration_rate)
    total_peak = len(peaks[peaks == 1])
    total_trough = len(troughs[troughs == 1])

    # calculate brv
    temp = rsp_signals.reset_index()
    d_peaks_in_ms = np.diff(temp[temp['RSP_Peaks'] == 1].index.values)
    d_peaks_in_ms = d_peaks_in_ms[~np.isnan(d_peaks_in_ms)] 
    
    d_d_peaks_in_ms = np.diff(d_peaks_in_ms)
    d_d_peaks_in_ms = d_d_peaks_in_ms[~np.isnan(d_d_peaks_in_ms)] 
    breath_rate_variability = np.sqrt(np.mean(np.square(d_d_peaks_in_ms)))

    
    # Calculate the areas for peak-to-trough and trough-to-peak transitions and average latency
    shaded_areas_peak_to_trough = []
    shaded_areas_trough_to_peak = []
    
    latency_peak_to_trough = []
    latency_trough_to_peak = []
    
    for i in range(1, len(cleaned_respiration)):
        if peaks[i] == 1 and i < len(troughs) - 1:
            # Find the next trough index after this peak
            next_trough_idx = np.where(troughs[i:] == 1)[0]
            if next_trough_idx.size > 0:
                next_trough_idx = next_trough_idx[0] + i
                area = np.trapz(cleaned_respiration.iloc[i:next_trough_idx], dx=1/sample_rate)
                shaded_areas_peak_to_trough.append(area)
                
                latency = (next_trough_idx - i) / sample_rate
                latency_peak_to_trough.append(latency)

                
        elif troughs[i] == 1 and i < len(peaks) - 1:
            # Find the next peak index after this trough
            next_peak_idx = np.where(peaks[i:] == 1)[0]
            if next_peak_idx.size > 0:
                next_peak_idx = next_peak_idx[0] + i
                area = np.trapz(cleaned_respiration.iloc[i:next_peak_idx], dx=1/sample_rate)
                shaded_areas_trough_to_peak.append(area)
                
                latency = (next_peak_idx - i) / sample_rate
                latency_trough_to_peak.append(latency)

                
    sum_peak_to_trough = np.sum(np.abs(shaded_areas_peak_to_trough))
    sum_trough_to_peak = np.sum(np.abs(shaded_areas_trough_to_peak))
    average_latency_peak_to_trough = np.mean(latency_peak_to_trough)
    average_latency_trough_to_peak = np.mean(latency_trough_to_peak)
    # Return the calculated metrics
    return {
        "Mean Respiration Rate": mean_respiration_rate,
        "Total Area Peak to Trough": sum_peak_to_trough,
        "Total Area Trough to Peak": sum_trough_to_peak,
        "Average time Peak to Trough": average_latency_peak_to_trough,
        "Average time Trough to Peak": average_latency_trough_to_peak,
        "Total_peaks": total_peak,
        "Total_trough": total_trough,
        "Breath_rate_variability" : breath_rate_variability,
    }


# In[14]:


def get_ecg_features(ecg, time_in_sec, fs):
    """
    Compute ECG features from raw ECG signal.

    Parameters
    ----------
    ecg : array-like
        Raw ECG signal.
    time_in_sec : array-like
        Timestamps corresponding to each sample of the ECG signal.
    fs : float
        Sampling frequency of the ECG signal.

    Returns
    -------
    array
        Array of ECG features: [mean heart rate, maximum heart rate, minimum heart rate, heart rate variability].
    """
    try:
        b, a = butter(4, (0.25, 25), 'bandpass', fs=fs)
        ecg_filt = filtfilt(b, a, ecg, axis=0)
        ecg_cleaned = nk.ecg_clean(ecg_filt, sampling_rate=fs)
        instant_peaks, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs,method="engzeemod2012")
    except Exception as e:
        raise ValueError("Error processing ECG signal: " + str(e))

    rr_times = time_in_sec[rpeaks['ECG_R_Peaks']]
    if len(rr_times) == 0:
        raise ValueError("No R-peaks detected in ECG signal.")
    
    # Assuming d_rr contains the time intervals between successive heartbeats in seconds
    d_rr = np.diff(rr_times)
    heart_rate = 60 / d_rr
    if heart_rate.size == 0:
        raise ValueError("Error computing heart rate from ECG signal.")
    
    valid_heart_rate = heart_rate[~np.isnan(heart_rate)]
    z_scores = np.abs(stats.zscore(valid_heart_rate))

    # Define a z-score threshold beyond which a value is considered an outlier
    z_score_threshold = 4.0

    # Remove outliers from the valid_heart_rate array
    heart_rate = valid_heart_rate[z_scores <= z_score_threshold]

    hr_mean = np.nanmean(heart_rate)
    hr_min = np.nanmin(heart_rate)
    hr_max = np.nanmax(heart_rate)
    d_rr_ms = 1000 * d_rr
    d_d_rr_ms = np.diff(d_rr_ms)

    valid_d_d_rr_ms = d_d_rr_ms[~np.isnan(d_d_rr_ms)] 
    z_scores = np.abs(stats.zscore(valid_d_d_rr_ms))
    d_d_rr_ms= valid_d_d_rr_ms[z_scores <= z_score_threshold]
    heart_rate_variability = np.sqrt(np.nanmean(np.square(d_d_rr_ms)))

    # Create a new signal 'ecg_with_rr_intervals' with RR intervals and a 1-second window around each RR interval
    ecg_with_rr_intervals = []
    ecg_with_rr_intervals_cleaned = []

    for rr_interval in rr_times:
        start_time = rr_interval - 0.1 # 1 second before the RR interval
        end_time = rr_interval + 0.1   # 1 second after the RR interval
        indices = np.where((time_in_sec >= start_time) & (time_in_sec <= end_time))[0]

        # Validate indices to ensure they are within bounds
        indices = indices[(indices >= 0) & (indices < len(ecg))]

        if len(indices) > 0:
            ecg_with_rr_intervals.extend(ecg[indices])
            ecg_with_rr_intervals_cleaned.extend(ecg_cleaned[indices])

    # Convert lists to NumPy arrays
    ecg_with_rr_intervals = np.array(ecg_with_rr_intervals)
    ecg_with_rr_intervals_cleaned = np.array(ecg_with_rr_intervals_cleaned)

    # Calculate noise power (mean squared amplitude of noise)
    signal_power = np.var(ecg_with_rr_intervals)
    noise_power = np.var(ecg_with_rr_intervals - ecg_with_rr_intervals_cleaned)

    # Calculate noise power (mean squared amplitude of noise)
    #signal_power = np.var(ecg)
    #noise_power = np.var(ecg - ecg_cleaned)

     # Calculate SNR in dB and append it to the array
    snr_values = 10 * np.log10(signal_power / noise_power)
    
    return hr_mean, hr_max, hr_min, heart_rate_variability, snr_values





CHUCK_SIZE = 30 * 250 #60 second * 250 Hz



# Function to split the data into chunks of a specified size
def split_into_chunks(group, chunk_size=CHUCK_SIZE):
    num_chunks = len(group) // chunk_size  # Calculate how many full chunks we can have
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append({
            'ecg': group['ecg'].iloc[start:end].tolist(),
            'resp': group['resp'].iloc[start:end].tolist()
        })
    return chunks

# Group the DataFrame by 'subject_id'
grouped = df.groupby('subject_id')

# Dictionary to hold the chunks for each subject
subject_chunks = {}

# Loop through each group, split into chunks, and store in the dictionary
for subject_id, group in grouped:
    subject_chunks[subject_id] = split_into_chunks(group)

# Flatten the dictionary into a DataFrame
# Initialize an empty list to hold all chunk records
chunk_data = []

# Iterate through each subject and their chunks
for subject_id, chunks in subject_chunks.items():
    for index, chunk in enumerate(chunks):
        chunk_data.append({
            'subject_id': subject_id,
            'chunk_id': index,
            'ecg': chunk['ecg'],
            'resp': chunk['resp']
        })

# Convert the list of dictionaries to a DataFrame
chunks_df = pd.DataFrame(chunk_data)

# Print the DataFrame structure
print(chunks_df.head())


# In[22]:


chunks_df


# In[23]:


len(chunks_df)


# In[24]:


chunks_df = chunks_df[chunks_df['resp'].apply(len) == CHUCK_SIZE]
chunks_df = chunks_df[chunks_df['ecg'].apply(len) == CHUCK_SIZE]


# In[25]:


len(chunks_df )


# In[26]:


chunks_df.head()


# In[ ]:





# In[27]:


SAMPLE_RATE = 250
CUT_OFF_SNR = 5

snr_list = []
# Pre-define new columns for calculated metrics
chunks_df['Mean_Respiration_Rate'] = np.nan
chunks_df['Total_Area_Peak_to_Trough'] = np.nan
chunks_df['Total_Area_Trough_to_Peak'] = np.nan
chunks_df['Mean_latency_Trough_to_Peak'] = np.nan
chunks_df['Mean_latency_Peak_to_Trough'] = np.nan
chunks_df['hr_min'] = np.nan
chunks_df['hr_mean'] = np.nan
chunks_df['hr_max'] = np.nan
chunks_df['hrv'] = np.nan
chunks_df['snr'] = np.nan
chunks_df['Total_peaks'] = np.nan
chunks_df['Total_trough'] = np.nan
chunks_df['Breath_rate_variability'] = np.nan

# List to track indices of rows that cause errors
error_indices = []

for idx, row in chunks_df.iterrows():
    print(idx)
        
    try:
        hr_mean, hr_max, hr_min, heart_rate_variability, snr_values = get_ecg_features(pd.Series(row['ecg']), np.arange(0,len(row['ecg'])/SAMPLE_RATE,1/SAMPLE_RATE), fs=SAMPLE_RATE)
        snr_list.append(snr_values)
        # if snr_values < CUT_OFF_SNR:
        #     continue
            
        # Try to calculate respiration metrics for the current row
        results = calculate_respiration_metrics(row['resp'])
        chunks_df.at[idx, 'Mean_Respiration_Rate'] = results['Mean Respiration Rate']
        chunks_df.at[idx, 'Total_Area_Trough_to_Peak'] = results['Total Area Trough to Peak']
        chunks_df.at[idx, 'Total_Area_Peak_to_Trough'] = results['Total Area Peak to Trough']
        chunks_df.at[idx, 'Mean_latency_Trough_to_Peak'] = results['Average time Trough to Peak']
        chunks_df.at[idx, 'Mean_latency_Peak_to_Trough'] = results['Average time Peak to Trough']
        chunks_df.at[idx, 'hr_min'] = hr_min
        chunks_df.at[idx, 'hr_mean'] = hr_mean
        chunks_df.at[idx, 'hr_max'] = hr_max
        chunks_df.at[idx, 'hrv'] = heart_rate_variability
        chunks_df.at[idx, 'snr'] = snr_values
        chunks_df.at[idx, 'Total_peaks'] = results['Total_peaks']
        chunks_df.at[idx, 'Total_trough'] = results['Total_trough']
        chunks_df.at[idx, 'Breath_rate_variability'] = results['Breath_rate_variability']
        
    except Exception as e:
        # Log error and mark index for potential removal
        print(f"Error processing row {idx}: {e}")
        error_indices.append(idx)

# Optionally, drop rows that caused errors from the DataFrame
chunks_df = chunks_df.drop(index=error_indices)

# Print updated DataFrame structure or save it to a file
print(chunks_df.head())


# In[28]:


len(chunks_df)


# In[29]:


chunks_df


# In[30]:


chunks_df = chunks_df.dropna(subset=['Mean_Respiration_Rate'])
chunks_df = chunks_df.dropna(subset=['Mean_latency_Trough_to_Peak'])
chunks_df = chunks_df.dropna(subset=['Mean_latency_Peak_to_Trough'])


# In[31]:


len(chunks_df)


# In[32]:


chunks_df.head()


# In[33]:


chunks_df.to_pickle(DATA_DIR + "fantasia_dataset_preprocessing.plk")




