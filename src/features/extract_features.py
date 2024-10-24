# Full Feature Extraction Code

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Paths to STFT data and features output
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stft')
features_output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'features')

# Ensure the features directory exists
if not os.path.exists(features_output_path):
    os.makedirs(features_output_path)

# Frequency band definitions (in Hz)
frequency_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# Sampling rate assumption for STFT analysis (modify if different)
sampling_rate = 256

# Function to calculate band power
def calculate_band_power(stft_matrix, freqs, band):
    """
    Calculate the average power in the given frequency band.
    """
    band_freqs = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if band_freqs.size == 0:
        print(f"Warning: No frequencies found in band {band}. Skipping band power calculation.")
        return 0.0
    band_power = np.mean(np.abs(stft_matrix[band_freqs, :] ** 2), axis=0)
    return np.mean(band_power)

# Function to calculate spectral entropy
def calculate_spectral_entropy(stft_matrix):
    """
    Calculate the spectral entropy of the given STFT data.
    """
    power_spectrum = np.abs(stft_matrix) ** 2
    total_power = np.sum(power_spectrum, axis=0)
    if np.any(total_power == 0):
        print("Warning: Some columns have total power of zero, skipping spectral entropy calculation for those columns.")
        power_spectrum[:, total_power == 0] = 1.0  # Avoid division by zero
    power_spectrum /= total_power  # Normalize
    return entropy(power_spectrum, base=2, axis=0).mean()

# Function to extract features from an STFT CSV file
def extract_features_from_file(file_path):
    df = pd.read_csv(file_path)
    # Extract signal values as NumPy array
    stft_data = df.to_numpy()

    # Assuming the first column contains frequency information
    frequencies = stft_data[:, 0]
    # The rest are STFT values
    stft_matrix = stft_data[:, 1:]

    # Calculate features for each frequency band
    features = {}
    for band_name, band_range in frequency_bands.items():
        features[f'{band_name}_power'] = calculate_band_power(stft_matrix, frequencies, band_range)

    # Calculate spectral entropy
    features['spectral_entropy'] = calculate_spectral_entropy(stft_matrix)

    return features

# Main function to extract features from all STFT files
def main():
    feature_list = []
    try:
        # List all files in the data path to ensure there are files to process
        files = os.listdir(data_path)
        if not files:
            print("No files found in the data path. Please check the path and try again.")
            return

        # Extract features from each file
        for file in files:
            if file.endswith('_stft.csv'):
                file_path = os.path.join(data_path, file)
                print(f"Processing file: {file}")
                features = extract_features_from_file(file_path)
                # Extract subject and condition from filename
                filename_parts = file.split('_')
                subject = filename_parts[0]
                condition = '_'.join(filename_parts[1:-1])
                # Add subject and condition to features
                features['subject'] = subject
                features['condition'] = condition
                feature_list.append(features)
    except FileNotFoundError:
        print(f"The directory {data_path} does not exist. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Save the extracted features if available
    if feature_list:
        features_df = pd.DataFrame(feature_list)
        features_output_file = os.path.join(features_output_path, 'features.csv')
        features_df.to_csv(features_output_file, index=False)
        print(f"Feature extraction complete. Features saved to: {features_output_file}")
    else:
        print("No features extracted. Please check the input files and try again.")

if __name__ == "__main__":
    main()


