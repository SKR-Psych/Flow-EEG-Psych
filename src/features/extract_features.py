# src/features/extract_features.py

import os
import pandas as pd
import numpy as np
from scipy.signal import welch

def extract_features_from_signal(signal, fs):
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['var'] = np.var(signal)
    features['skew'] = pd.Series(signal).skew()
    features['kurtosis'] = pd.Series(signal).kurtosis()
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(signal)

    freqs, psd = welch(signal, fs)
    total_power = np.trapz(psd, freqs)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.trapz(psd[idx], freqs[idx])
        features[f'bandpower_{band_name}'] = band_power
        features[f'relative_power_{band_name}'] = band_power / total_power if total_power > 0 else 0

    return features

def process_file(signal_file_path, metadata_file_path, fs):
    try:
        signals_df = pd.read_csv(signal_file_path)
        metadata_df = pd.read_csv(metadata_file_path)

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

        for _, event in metadata_df.iterrows():
            event_type = event['type']
            init_time = event['init_time']
            start_idx = int(init_time * fs)
            window_size = int(2 * fs)
            end_idx = start_idx + window_size

            if end_idx > len(signals_df):
                end_idx = len(signals_df)
            if end_idx - start_idx < fs:
                continue

            for channel in available_eeg_channels:
                signal = signals_df[channel].values[start_idx:end_idx]
                if len(signal) < fs:
                    continue

                features = extract_features_from_signal(signal, fs)
                features['channel'] = channel
                features['event_type'] = event_type
                features['participant_id'] = os.path.basename(signal_file_path).split('_')[0]
                features['condition'] = '_'.join(os.path.basename(signal_file_path).split('_')[1:-1])
                features_list.append(features)

        if not features_list:
            print(f"No features extracted from {signal_file_path}")
            return None

        features_df = pd.DataFrame(features_list)
        return features_df

    except Exception as e:
        print(f"Error processing {signal_file_path}: {e}")
        return None

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(current_dir, '../../data/processed/')
    features_output_dir = os.path.join(current_dir, '../../data/features/')

    os.makedirs(features_output_dir, exist_ok=True)
    fs = 256  # Adjust based on your data

    signal_files = [f for f in os.listdir(processed_data_dir) if f.endswith('_signals.csv')]
    all_features = []

    for signal_file in signal_files:
        signal_file_path = os.path.join(processed_data_dir, signal_file)
        metadata_file = signal_file.replace('_signals.csv', '_metadata.csv')
        metadata_file_path = os.path.join(processed_data_dir, metadata_file)

        if not os.path.exists(metadata_file_path):
            print(f"Metadata file not found for {signal_file}")
            continue

        print(f'Processing {signal_file}')
        features_df = process_file(signal_file_path, metadata_file_path, fs)

        if features_df is not None:
            all_features.append(features_df)

    if all_features:
        all_features_df = pd.concat(all_features, ignore_index=True)
        all_features_df.to_csv(os.path.join(features_output_dir, 'all_features.csv'), index=False)
        print('Saved all features to all_features.csv')
    else:
        print("No features were extracted.")

if __name__ == '__main__':
    main()
