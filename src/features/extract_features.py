import os
import pandas as pd
import numpy as np
from scipy.signal import welch
import json

# Directory containing processed EEG CSV files
processed_data_dir = r'C:\Users\Sami\Desktop\Uni\Flow-EEG-Psych\data\processed'

# Define frequency bands for EEG analysis
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Default sampling rate (used if not available in metadata)
default_fs = 256  # Sampling rate in Hz

# Function to compute band power for a specific frequency band
def compute_band_power(signal, fs, band):
    fmin, fmax = band
    nperseg = min(256, len(signal))  # Adjust nperseg based on signal length
    f, Pxx = welch(signal, fs, nperseg=nperseg)
    band_power = np.trapz(Pxx[(f >= fmin) & (f <= fmax)], f[(f >= fmin) & (f <= fmax)])
    return band_power

# Function to extract features from a CSV file containing EEG data
def extract_features(file_path, fs):
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
            signal = df_signal[column].values
            if len(signal) < 2:
                print(f"Warning: Signal in {column} is too short to analyze.")
                band_power = np.nan
            else:
                band_power = compute_band_power(signal, fs, band_range)
            feature_name = f"{column}_{band_name}_power"
            features[feature_name] = band_power
    
    return features

# Main function to extract features from all files and save to output
def main():
    # Directory to save extracted features
    output_dir = r'C:\Users\Sami\Desktop\Uni\Flow-EEG-Psych\data\features'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all processed EEG CSV files and extract features
    all_features = []
    for file in os.listdir(processed_data_dir):
        if file.endswith('_signals.csv'):
            file_path = os.path.join(processed_data_dir, file)
            try:
                # Attempt to read sampling rate from metadata if available
                metadata_path = file_path.replace('_signals.csv', '_metadata.csv')
                if os.path.exists(metadata_path):
                    metadata = pd.read_csv(metadata_path)
                    if 'sampling_rate' in metadata.columns:
                        fs_raw = metadata['sampling_rate'].values[0]
                        # Handle nested list format
                        fs = fs_raw if isinstance(fs_raw, (int, float)) else float(fs_raw[0][0])
                    else:
                        fs = default_fs
                else:
                    fs = default_fs

                features = extract_features(file_path, fs)
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
        'sampling_rate': default_fs,
        'description': "EEG frequency band power features extracted from raw signals"
    }
    metadata_json_path = os.path.join(output_dir, 'features_metadata.json')
    with open(metadata_json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_json_path}")

if __name__ == "__main__":
    main()

