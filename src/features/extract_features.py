# src/features/extract_features.py

import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne.channels import make_standard_montage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from tqdm import tqdm

# Define Constants
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'features')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'decoders')

# Ensure output directories exist
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Sampling Frequency
SFREQ = 256  # Replace with your actual sampling frequency if different

# Define EEG Channels (first 56 columns based on your sample)
EEG_CHANNELS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
    'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
    'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
    'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
    'P10', 'PO8', 'PO4', 'O2'
]

# Frequency Bands for FBCSP
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 15),
    'beta': (15, 24),
    'gamma': (24, 50)
}

N_COMPONENTS = 3  # Number of CSP components per band

class FBCSPExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_bands, n_components=3):
        self.frequency_bands = frequency_bands
        self.n_components = n_components
        self.csp_pipelines = {}
        self.filter_params = {}  # To store filter parameters for each band

    def fit(self, X, y):
        for band_name, (fmin, fmax) in self.frequency_bands.items():
            # Bandpass filter
            filt = mne.filter.create_filter(
                X.mean(axis=0),
                SFREQ,
                l_freq=fmin,
                h_freq=fmax,
                method='firwin',
                verbose=False
            )
            # Apply filter to all epochs
            X_filtered = mne.filter.filter_data(X, SFREQ, l_freq=fmin, h_freq=fmax, verbose=False)
            
            # Initialize CSP
            csp = mne.decoding.CSP(n_components=self.n_components, 
                                   reg=None, 
                                   log=True, 
                                   norm_trace=False)
            # Fit CSP
            csp.fit(X_filtered, y)
            self.csp_pipelines[band_name] = csp
            self.filter_params[band_name] = (fmin, fmax)
        return self

    def transform(self, X):
        features = []
        for band_name, csp in self.csp_pipelines.items():
            # Bandpass filter using stored filter params
            fmin, fmax = self.filter_params[band_name]
            X_filtered = mne.filter.filter_data(X, SFREQ, l_freq=fmin, h_freq=fmax, verbose=False)
            # Apply CSP
            csp_features = csp.transform(X_filtered)
            features.append(csp_features)
        # Concatenate all band features
        return np.hstack(features)

def preprocess_eeg(raw):
    """
    Preprocess raw EEG data: filtering and artifact removal using ICA.
    """
    # Set EEG montage
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Bandpass filter
    raw.filter(0.5, 50., fir_design='firwin', verbose=False)
    
    # Apply ICA for artifact removal
    ica = ICA(n_components=15, random_state=97, max_iter='auto', verbose=False)
    ica.fit(raw)
    
    # Find and exclude EOG artifacts
    eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
    ica.exclude = eog_indices
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    
    return raw_clean

def extract_epochs(signals_df, metadata_df, sfreq=SFREQ, tmin=0, tmax=2):
    """
    Extract epochs from the continuous EEG data based on event timings.
    """
    epochs = []
    labels = []
    participant_id = os.path.basename(signals_df.attrs['filename']).replace('_signals.csv', '')
    
    for idx, event in metadata_df.iterrows():
        event_time = event['init_time']  # in seconds
        label = event['urevent']  # Assuming 'urevent' is the label
        
        # Convert event time to sample index
        sample = int(event_time * sfreq)
        
        # Define epoch start and end samples
        start_sample = sample
        end_sample = sample + int(tmax * sfreq)
        
        # Check for boundaries
        if end_sample > len(signals_df):
            print(f"Epoch end sample {end_sample} exceeds data length {len(signals_df)}. Skipping epoch.")
            continue
        
        # Extract EEG data for the epoch
        epoch_data = signals_df.iloc[start_sample:end_sample][EEG_CHANNELS].values.T  # shape: (n_channels, n_times)
        
        epochs.append(epoch_data)
        labels.append(label)
    
    return np.array(epochs), np.array(labels)

def main():
    print("Starting feature extraction...")
    
    # Initialize Feature Extractor
    fbcsp = FBCSPExtractor(frequency_bands=FREQUENCY_BANDS, n_components=N_COMPONENTS)
    
    # Initialize Lists to Store Data
    all_epochs = []
    all_labels = []
    participant_ids = []
    
    # Get list of signals CSV files
    signals_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_signals.csv')]
    
    if not signals_files:
        print("No signals CSV files found in the processed data directory.")
        return
    
    for signals_file in tqdm(signals_files, desc='Processing Signals Files'):
        # Derive corresponding metadata file name
        base_name = signals_file.replace('_signals.csv', '')
        metadata_file = f"{base_name}_metadata.csv"
        signals_path = os.path.join(PROCESSED_DATA_DIR, signals_file)
        metadata_path = os.path.join(PROCESSED_DATA_DIR, metadata_file)
        
        if not os.path.exists(metadata_path):
            print(f"Metadata file {metadata_file} not found for signals file {signals_file}. Skipping.")
            continue
        
        # Read signals CSV
        try:
            signals_df = pd.read_csv(signals_path)
            # Optional: Store filename in DataFrame attributes for tracking
            signals_df.attrs['filename'] = signals_file
        except Exception as e:
            print(f"Error reading {signals_file}: {e}")
            continue
        
        # Read metadata CSV
        try:
            metadata_df = pd.read_csv(metadata_path)
        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")
            continue
        
        # Extract epochs and labels
        epochs, labels = extract_epochs(signals_df, metadata_df, sfreq=SFREQ, tmin=0, tmax=2)
        
        if len(epochs) == 0:
            print(f"No valid epochs extracted from {signals_file}.")
            continue
        
        all_epochs.append(epochs)
        all_labels.append(labels)
        participant_ids.extend([base_name] * len(labels))
    
    if not all_epochs:
        print("No epochs extracted from any files. Exiting.")
        return
    
    # Concatenate all epochs and labels
    X_all = np.concatenate(all_epochs, axis=0)  # Shape: (total_epochs, n_channels, n_times)
    y_all = np.concatenate(all_labels, axis=0)  # Shape: (total_epochs,)
    participant_ids = np.array(participant_ids)
    
    # Shuffle Data
    X_all, y_all, participant_ids = shuffle(X_all, y_all, participant_ids, random_state=42)
    
    # Preprocess EEG Data (filtering and artifact removal)
    # Note: Since epochs are already extracted and filtering is applied in preprocess_eeg,
    # you might skip additional filtering here unless necessary.
    # For demonstration, we'll proceed with FBCSP directly.
    
    # Fit FBCSP
    print("Fitting FBCSP...")
    fbcsp.fit(X_all, y_all)
    X_features = fbcsp.transform(X_all)  # Shape: (n_epochs, n_features)
    
    # Standardize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Save Scaler and FBCSP Models
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    fbcsp_path = os.path.join(MODEL_DIR, 'fbcsp.joblib')
    joblib.dump(scaler, scaler_path)
    joblib.dump(fbcsp, fbcsp_path)
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved FBCSP extractor to {fbcsp_path}")
    
    # Save Features and Labels
    features_df = pd.DataFrame(X_scaled)
    features_df['label'] = y_all
    features_df['participant_id'] = participant_ids
    features_csv_path = os.path.join(FEATURES_DIR, 'all_features.csv')
    features_df.to_csv(features_csv_path, index=False)
    print(f"Saved all features to {features_csv_path}")
    
    print("Feature extraction completed successfully.")

if __name__ == '__main__':
    main()

