# src/features/extract_features.py

import os
import numpy as np
import pandas as pd
import mne
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from load_data import load_mat_file, get_mat_file_list

SFREQ = 256  # Define the sampling frequency here (example value)

class FBCSPExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_bands, n_components=3):
        self.frequency_bands = frequency_bands
        self.n_components = n_components
        self.csp_pipelines = {}
        self.filter_params = {}

    def fit(self, X, y):
        for band_name, (fmin, fmax) in self.frequency_bands.items():
            print(f"Fitting CSP for {band_name} band: {fmin}-{fmax} Hz")
            
            # Convert X to float64 for MNE's filter function
            X_filtered = mne.filter.filter_data(X.astype(np.float64), SFREQ, l_freq=fmin, h_freq=fmax, verbose=False)
            
            # Initialize and fit CSP
            csp = mne.decoding.CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
            csp.fit(X_filtered, y)
            self.csp_pipelines[band_name] = csp
            self.filter_params[band_name] = (fmin, fmax)
        
        return self

    def transform(self, X):
        features = []
        for band_name, csp in self.csp_pipelines.items():
            fmin, fmax = self.filter_params[band_name]
            
            # Filter X for this frequency band and transform with CSP
            X_filtered = mne.filter.filter_data(X.astype(np.float64), SFREQ, l_freq=fmin, h_freq=fmax, verbose=False)
            X_csp = csp.transform(X_filtered)
            features.append(X_csp)
        
        return np.concatenate(features, axis=1)

def extract_epochs(signals_df, metadata_df, sfreq, tmin, tmax):
    epochs = []
    labels = []

    for _, event in metadata_df.iterrows():
        event_time = event['init_time']
        label = event['type']

        start_sample = int((event_time + tmin) * sfreq)
        end_sample = int((event_time + tmax) * sfreq)

        if start_sample < 0 or end_sample > len(signals_df):
            print(f"Epoch end sample {end_sample} exceeds data length {len(signals_df)}. Skipping epoch.")
            continue

        epoch = signals_df.iloc[start_sample:end_sample].values
        epochs.append(epoch)
        labels.append(label)

    return np.array(epochs), labels

def main():
    print("Starting feature extraction...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(current_dir, '../../data/processed/')
    os.makedirs(processed_data_dir, exist_ok=True)

    signal_files = [f for f in os.listdir(processed_data_dir) if f.endswith('_signals.csv')]
    metadata_files = [f for f in os.listdir(processed_data_dir) if f.endswith('_metadata.csv')]

    all_epochs = []
    all_labels = []

    for signal_file, metadata_file in zip(signal_files, metadata_files):
        print(f"Processing {signal_file}...")

        signals_df = pd.read_csv(os.path.join(processed_data_dir, signal_file))
        metadata_df = pd.read_csv(os.path.join(processed_data_dir, metadata_file))

        epochs, labels = extract_epochs(signals_df, metadata_df, sfreq=SFREQ, tmin=0, tmax=2)
        all_epochs.extend(epochs)
        all_labels.extend(labels)

    X_all = np.array(all_epochs, dtype=np.float32)
    y_all = np.array(all_labels)

    frequency_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30)
    }

    fbcsp = FBCSPExtractor(frequency_bands=frequency_bands, n_components=3)
    fbcsp.fit(X_all, y_all)
    features = fbcsp.transform(X_all)

    feature_df = pd.DataFrame(features)
    feature_csv_path = os.path.join(processed_data_dir, 'extracted_features.csv')
    feature_df.to_csv(feature_csv_path, index=False)
    print(f'Features saved to {feature_csv_path}')

if __name__ == "__main__":
    main()



