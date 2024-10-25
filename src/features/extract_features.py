# src/features/extract_features.py

import os
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

def extract_features_from_signal(signal, fs):
    """
    Extract features from a single EEG signal.

    Parameters:
    - signal: 1D numpy array of the EEG signal.
    - fs: Sampling frequency.

    Returns:
    - features: Dictionary of extracted features.
    """
    features = {}

    # Check if the signal contains enough data
    if len(signal) < 1:
        print("Warning: Signal length is zero.")
        return features

    # Time-domain features
    try:
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['skew'] = skew(signal)
        features['kurtosis'] = kurtosis(signal)
    except Exception as e:
        print(f"Error computing time-domain features: {e}")

    # Frequency-domain features
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 1024))

        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        total_power = np.trapz(psd, freqs)

        for band_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psd[idx], freqs[idx])
            features[f'bandpower_{band_name}'] = band_power
            # Relative band power
            features[f'relative_bandpower_{band_name}'] = band_power / total_power if total_power != 0 else 0
    except Exception as e:
        print(f"Error computing frequency-domain features: {e}")

    return features

def process_file(signal_file_path, metadata_file_path, fs):
    """
    Process a single signal file to extract features.

    Parameters:
    - signal_file_path: Path to the signal CSV file.
    - metadata_file_path: Path to the metadata CSV file.
    - fs: Sampling frequency.

    Returns:
    - features_df: DataFrame containing features for each channel.
    """
    try:
        # Load signal data
        signals_df = pd.read_csv(signal_file_path)
    except Exception as e:
        print(f"Error loading signal file {signal_file_path}: {e}")
        return pd.DataFrame()

    # Load metadata if available
    if os.path.exists(metadata_file_path):
        try:
            metadata_df = pd.read_csv(metadata_file_path)
        except Exception as e:
            print(f"Error loading metadata file {metadata_file_path}: {e}")
            metadata_df = None
    else:
        metadata_df = None

    # Identify EEG channels (adjust as necessary)
    eeg_channels = [
        'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
        'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
        'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
        'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
        'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
        'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
        'P10', 'PO8', 'PO4', 'O2'
    ]

    available_eeg_channels = [ch for ch in eeg_channels if ch in signals_df.columns]

    features_list = []
    for channel in available_eeg_channels:
        signal = signals_df[channel].values

        # Handle missing values in the signal
        if np.isnan(signal).any():
            print(f"Missing values found in channel {channel}. Applying interpolation.")
            signal = pd.Series(signal).interpolate().fillna(method='bfill').fillna(method='ffill').values

        # Extract features from the signal
        features = extract_features_from_signal(signal, fs)
        features['channel'] = channel
        features['file'] = os.path.basename(signal_file_path)

        # Optional: Add metadata information
        if metadata_df is not None and not metadata_df.empty:
            # For example, include the number of events or specific event types
            features['num_events'] = len(metadata_df)
            # You can customize this part based on your metadata structure

        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    return features_df

def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(current_dir, '../../data/processed/')
    features_output_dir = os.path.join(current_dir, '../../data/features/')

    os.makedirs(features_output_dir, exist_ok=True)

    # Sampling frequency (adjust based on your data)
    fs = 256  # Hz

    # List all processed signal CSV files
    signal_files = [f for f in os.listdir(processed_data_dir) if f.endswith('_signals.csv')]

    if not signal_files:
        print("No processed signal files found.")
        return

    all_features = []

    for signal_file in signal_files:
        signal_file_path = os.path.join(processed_data_dir, signal_file)
        metadata_file = signal_file.replace('_signals.csv', '_metadata.csv')
        metadata_file_path = os.path.join(processed_data_dir, metadata_file)

        print(f'Processing {signal_file}')

        try:
            features_df = process_file(signal_file_path, metadata_file_path, fs)

            if features_df.empty:
                print(f"No features extracted for {signal_file}")
                continue

            # Save features for this file
            feature_file_name = signal_file.replace('_signals.csv', '_features.csv')
            feature_file_path = os.path.join(features_output_dir, feature_file_name)
            features_df.to_csv(feature_file_path, index=False)
            print(f'Saved features to {feature_file_path}')

            all_features.append(features_df)
        except Exception as e:
            print(f"Error processing {signal_file}: {e}")
            continue

    # Optionally, concatenate all features into a single DataFrame
    if all_features:
        all_features_df = pd.concat(all_features, ignore_index=True)
        all_features_df.to_csv(os.path.join(features_output_dir, 'all_features.csv'), index=False)
        print('Saved all features to all_features.csv')
    else:
        print("No features were extracted from any files.")

if __name__ == '__main__':
    main()

