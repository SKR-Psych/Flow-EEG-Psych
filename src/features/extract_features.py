import os
import pandas as pd
import numpy as np
from scipy.signal import welch
import json

# Directory containing processed EEG CSV files
processed_data_dir = "data/processed/"

# Define frequency bands for EEG analysis
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Sampling rate (should be extracted from metadata if available, this is an example value)
fs = 256  # Sampling rate in Hz

# Function to compute band power for a specific frequency band
def compute_band_power(signal, fs, band):
    fmin, fmax = band
    f, Pxx = welch(signal, fs, nperseg=1024)
    band_power = np.trapz(Pxx[(f >= fmin) & (f <= fmax)], f[(f >= fmin) & (f <= fmax)])
    return band_power

# Function to extract features from a CSV file containing EEG data
def extract_features(file_path):
    # Load the EEG data from CSV
    df_signal = pd.read_csv(file_path)
    
    # Extract subject and condition from the filename
    filename = os.path.basename(file_path)
    file_parts = filename.split('_')
    subject = file_parts[0]  # e.g., "S01"
    condition = '_'.join(file_parts[1:]).replace('_signals.csv', '')
    
    # Extract frequency domain features for each channel
    features = {
        'subject': subject,
        'condition': condition
    }
    
    for band_name, band_range in bands.items():
        for column in df_signal.columns:
            band_power = compute_band_power(df_signal[column].values, fs, band_range)
            feature_name = f"{column}_{band_name}_power"
            features[feature_name] = band_power
    
    return features

# Directory to save extracted features
output_dir = "data/features/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all processed EEG CSV files and extract features
all_features = []
for file in os.listdir(processed_data_dir):
    if file.endswith('_signals.csv'):
        file_path = os.path.join(processed_data_dir, file)
        try:
            features = extract_features(file_path)
            all_features.append(features)
            print(f"Extracted features from {file}")
        except Exception as e:
            print(f"Error extracting features from {file}: {e}")

# Save all extracted features to a CSV file
features_df = pd.DataFrame(all_features)
features_csv_path = os.path.join(output_dir, 'eeg_features.csv')
features_df.to_csv(features_csv_path, index=False)
print(f"All features saved to {features_csv_path}")

# Save metadata to a JSON file for reference
metadata = {
    'bands': bands,
    'sampling_rate': fs,
    'description': "EEG frequency band power features extracted from raw signals"
}
metadata_json_path = os.path.join(output_dir, 'features_metadata.json')
with open(metadata_json_path, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {metadata_json_path}")

