#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('jupyter nbconvert --to script ecg_respiratory_analysis_and_generate_preprocessing.ipynb')


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


# In[3]:


RUN = 'simple_mlp/'
CURRENT_DIR = os.getcwd() + '/..'
DATA_DIR = CURRENT_DIR + '/dataset/'
MASTER_OUTPUT_DIR = CURRENT_DIR + '/output/'
OUTPUT_DIR = MASTER_OUTPUT_DIR + RUN
print(DATA_DIR)
print(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR,exist_ok=True)


# In[4]:


# Load the dataset
file_path = DATA_DIR + 'fantasia_dataset_preprocessing.plk'
chunks_df = pd.read_pickle(file_path)


# Drop rows where 'Mean Respiration Rate' is NaN
chunks_df = chunks_df.dropna(subset=['Mean_Respiration_Rate'])

# Convert 'ecg' column to a 3D numpy array
X = np.array(chunks_df['ecg'].tolist())
y = chunks_df['Mean_Respiration_Rate'].values

# Normalize each ECG chunk individually and handle NaNs
X_normalized = np.zeros_like(X)
for i in range(X.shape[0]):
    mean = np.mean(X[i])
    std = np.std(X[i])
    if std == 0:  # Avoid division by zero
        std = 1
    X_normalized[i] = (X[i] - mean) / std

# Identify rows with NaNs in the normalized data
nan_rows = np.isnan(X_normalized).any(axis=1)

# Remove rows with NaNs
X_normalized = X_normalized[~nan_rows]
y = y[~nan_rows]
cleaned_chunks_df = chunks_df[~nan_rows]
# Check again for NaN values in normalized data
print(f"NaN in X_normalized after cleaning: {np.isnan(X_normalized).sum()}, Inf in X_normalized: {np.isinf(X_normalized).sum()}")


# In[44]:


# Calculate the 2.5th and 97.5th percentiles
lower_percentile = np.percentile(y, 0.5)
upper_percentile = np.percentile(y, 99.5)



# Verify the clipping
print(f"1th percentile: {lower_percentile}")
print(f"99th percentile: {upper_percentile}")
print(f"Before clipping: min = {y.min()}, max = {y.max()}")
# Clip the values in y_train
y = np.clip(y, lower_percentile, upper_percentile)
print(f"After clipping: min = {y.min()}, max = {y.max()}")

# Continue with training the model using y_train_clipped


# In[45]:


from sklearn.model_selection import train_test_split

# Split the data by subjects
subjects = cleaned_chunks_df['subject_id'].unique()

# First, split subjects into training + validation, and test subjects
train_val_subjects, test_subjects = train_test_split(subjects, test_size=0.25, random_state=42)

# Now split the training + validation subjects to create a training set and validation set
train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15, random_state=42)

# Create boolean masks for train, validation, and test subjects
train_mask = cleaned_chunks_df['subject_id'].isin(train_subjects)
val_mask = cleaned_chunks_df['subject_id'].isin(val_subjects)
test_mask = cleaned_chunks_df['subject_id'].isin(test_subjects)

# Use these masks to create the actual train, validation, and test datasets
train_data = cleaned_chunks_df[train_mask]
val_data = cleaned_chunks_df[val_mask]
test_data = cleaned_chunks_df[test_mask]

# Now you have train, validation, and test datasets based on the subjects


# In[46]:


# Apply the masks to create train, validation, and test sets
X_train = np.expand_dims(X_normalized[train_mask], 2)
y_train = y[train_mask] 
X_val = np.expand_dims(X_normalized[val_mask], 2)
y_val = y[val_mask]
X_test = np.expand_dims(X_normalized[test_mask], 2)
y_test = y[test_mask]

# Print shapes of the datasets to verify
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"y_test shape: {y_test.shape}")


# In[47]:


# Plot the first few samples of normalized ECG data in separate subplots
num_samples_to_plot = 7  # Number of samples to plot
fig, axs = plt.subplots(num_samples_to_plot, 1, figsize=(12, 10))

for i in range(num_samples_to_plot):
    axs[i].plot(X_normalized[i], label=f'Sample {i}')
    axs[i].set_title(f'Normalized ECG Sample {i}')
    axs[i].set_xlabel('Time (samples)')
    axs[i].set_ylabel('Normalized Amplitude')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'normalized_sample.png')
plt.show()


# In[48]:


len(y_test)


# In[49]:


len(y_train)


# In[ ]:





# In[50]:


import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the model
model = tf.keras.Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1)  # Output layer for regression
])


# In[51]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Compile the model with a smaller learning rate
model.compile(optimizer=Adam(learning_rate=0.00001), loss='mean_absolute_error', metrics=['mae'])

# Print the model summary
model.summary()

# Train the model with the validation data
history = model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_val, y_val))

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model with MSE and MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")


# In[52]:


# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig(OUTPUT_DIR + 'plot_loss.png')
plt.show()

# Plot training & validation MAE values
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig(OUTPUT_DIR + 'plot_loss_accuracy.png')
plt.show()


# In[53]:


from scipy.stats import pearsonr, linregress

# Calculate the Pearson correlation coefficient
correlation, p_value = pearsonr(y_test, y_pred.flatten())
print(f"Pearson Correlation: {correlation}")
print(f"P-value: {p_value}")

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred.flatten(), alpha=0.6, edgecolors='w', linewidth=0.5, label='Data points')

# Calculate the regression line
slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred.flatten())
regression_line = slope * y_test + intercept
plt.plot(y_test, regression_line, color='red', linewidth=2, label='Regression line')

plt.title('Actual vs Predicted Mean Respiration Rates')
plt.xlabel('Actual Mean Respiration Rate')
plt.ylabel('Predicted Mean Respiration Rate')
plt.grid(True)
plt.legend()

# Add text box with correlation and p-value
textstr = f'Pearson Correlation: {correlation:.2f}\nP-value: {p_value:.2e}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.25, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
               verticalalignment='top', bbox=props)
plt.savefig(OUTPUT_DIR + 'predict_vs_true.png')
plt.show()


# In[54]:


# Plot a histogram of the predicted mean respiration rates
plt.figure(figsize=(10, 6))
plt.hist(y_pred, bins=30, edgecolor='black')
plt.title('Histogram of Predicted Mean Respiration Rates')
plt.xlabel('Predicted Mean Respiration Rate')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(OUTPUT_DIR + 'predict_distribution.png')
plt.show()


# In[55]:


# Predict on the test set
y_pred = model.predict(X_test)

# Shuffle and select a few examples from the test set
num_examples_to_plot = 5
indices = np.arange(X_test.shape[0])
np.random.shuffle(indices)
selected_indices = indices[:num_examples_to_plot]

# Plot the selected examples from the test set
fig, axs = plt.subplots(num_examples_to_plot, 1, figsize=(12, 15))

for i, idx in enumerate(selected_indices):
    axs[i].plot(X_test[idx].flatten(), label='ECG Signal')
    axs[i].set_title(f'ECG Signal - Test Sample {idx}')
    axs[i].set_xlabel('Time (samples)')
    axs[i].set_ylabel('Normalized Amplitude')
    axs[i].legend(loc='upper right')
    axs[i].text(0.5, 0.95, f'Actual Mean Respiration Rate: {y_test[idx]:.2f}', horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
    axs[i].text(0.5, 0.85, f'Predicted Mean Respiration Rate: {y_pred[idx][0]:.2f}', horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'ecg_samples_prediction_example.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




