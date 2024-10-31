# src/features/extract_features.py

import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define Constants
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'features')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'decoders')
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Sampling Frequency (Replace with actual if different)
SFREQ = 256

# Define EEG Channels (Ensure this matches your data)
EEG_CHANNELS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
    'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
    'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
    'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
    'P10', 'PO8', 'PO4', 'O2'
]

# Frequency Bands for FBCSP (Adjusted)
FREQUENCY_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 15),
    'beta': (15, 24)
}

N_COMPONENTS = 2  # Reduced number of CSP components per band

# Define Paths
SIGNALS_DIR = PROCESSED_DATA_DIR
METADATA_DIR = PROCESSED_DATA_DIR
FEATURES_HDF5_PATH = os.path.join(FEATURES_DIR, 'flow_features.h5')

# Custom FBCSP Extractor
class FBCSPExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_bands, n_components=3, sfreq=256):
        self.frequency_bands = frequency_bands
        self.n_components = n_components
        self.sfreq = sfreq
        self.csp_pipelines = {}
        self.filter_params = {}
    
    def fit(self, X, y):
        for band_name, (fmin, fmax) in self.frequency_bands.items():
            print(f"Fitting CSP for {band_name} band: {fmin}-{fmax} Hz")
            
            # Verify data type and shape
            print(f"Data dtype: {X.dtype}, Data shape: {X.shape}, Is real: {np.isrealobj(X)}")
            
            # Ensure data is float64
            if X.dtype != np.float64:
                print(f"Converting data from {X.dtype} to float64.")
                X = X.astype(np.float64)
            
            # Check for NaNs or Infs
            if not np.isfinite(X).all():
                raise ValueError("Data contains NaNs or Infs.")
            
            # Filter data for the current band
            X_filtered = mne.filter.filter_data(X, self.sfreq, l_freq=fmin, h_freq=fmax, verbose=False)
            
            # Initialize and fit CSP
            csp = mne.decoding.CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
            csp.fit(X_filtered, y)
            
            # Store CSP pipeline
            self.csp_pipelines[band_name] = csp
            self.filter_params[band_name] = (fmin, fmax)
        return self
    
    def transform(self, X):
        features = []
        for band_name, csp in self.csp_pipelines.items():
            fmin, fmax = self.filter_params[band_name]
            # Ensure data is float64
            if X.dtype != np.float64:
                print(f"Converting data from {X.dtype} to float64 for transform.")
                X = X.astype(np.float64)
            # Filter data for the current band
            X_filtered = mne.filter.filter_data(X, self.sfreq, l_freq=fmin, h_freq=fmax, verbose=False)
            # Apply CSP
            csp_features = csp.transform(X_filtered)
            features.append(csp_features)
        # Concatenate features from all bands
        return np.hstack(features)

# Feature Extraction Function
def extract_features_from_file(signals_file, metadata_file):
    try:
        # Load Signals
        signals_path = os.path.join(SIGNALS_DIR, signals_file)
        signals_df = pd.read_csv(signals_path)
        signals_df = signals_df[EEG_CHANNELS].astype(np.float32)  # Use float32 to save memory
        
        # Load Metadata
        metadata_path = os.path.join(METADATA_DIR, metadata_file)
        metadata_df = pd.read_csv(metadata_path)
        
        # Extract Epochs and Labels
        epochs, labels = extract_epochs(signals_df, metadata_df, sfreq=SFREQ, tmin=0, tmax=2)
        if len(epochs) == 0:
            return None, None
        
        return epochs, labels
    except Exception as e:
        print(f"Error processing {signals_file} and {metadata_file}: {e}")
        return None, None

# Epoch Extraction Function
def extract_epochs(signals_df, metadata_df, sfreq=SFREQ, tmin=0, tmax=2):
    epochs = []
    labels = []
    skipped_events = 0
    total_events = len(metadata_df)
    
    for idx, event in metadata_df.iterrows():
        event_time = event.get('init_time', np.nan)  # in seconds
        label = event.get('urevent', np.nan)        # Assuming 'urevent' is the label
        
        # Check for NaN in event_time or label
        if pd.isna(event_time) or pd.isna(label):
            skipped_events += 1
            continue
        
        # Convert event time to sample index
        try:
            sample = int(event_time * sfreq)
        except (ValueError, TypeError):
            skipped_events += 1
            continue
        
        # Define epoch start and end samples
        start_sample = sample
        end_sample = sample + int(tmax * sfreq)
        
        # Check for boundaries
        if end_sample > len(signals_df):
            skipped_events += 1
            continue
        
        # Extract EEG data for the epoch
        epoch_data = signals_df.iloc[start_sample:end_sample].values.T  # shape: (n_channels, n_times)
        
        epochs.append(epoch_data.astype(np.float32))  # Use float32 to reduce memory
        labels.append(label)
    
    print(f"Total events: {total_events}, Skipped events: {skipped_events}, Extracted epochs: {len(epochs)}")
    return np.array(epochs), np.array(labels)

# Main Feature Extraction Function
def main():
    print("Starting feature extraction...")
    
    # Initialize FBCSP Extractor
    fbcsp = FBCSPExtractor(frequency_bands=FREQUENCY_BANDS, n_components=N_COMPONENTS, sfreq=SFREQ)
    
    # Prepare HDF5 file for storing features
    with h5py.File(FEATURES_HDF5_PATH, 'w') as h5f:
        h5f.create_group('features')
        h5f.create_group('labels')
        h5f.create_group('participant_ids')
    
    # Get list of processed files
    processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_signals.csv')]
    
    if not processed_files:
        print("No processed signal files found.")
        return
    
    # Pair signals and metadata files
    file_pairs = []
    for signals_file in processed_files:
        base_name = signals_file.replace('_signals.csv', '')
        metadata_file = f"{base_name}_metadata.csv"
        if os.path.exists(os.path.join(PROCESSED_DATA_DIR, metadata_file)):
            file_pairs.append((signals_file, metadata_file))
        else:
            print(f"Metadata file {metadata_file} not found for signals file {signals_file}. Skipping.")
    
    if not file_pairs:
        print("No valid file pairs found.")
        return
    
    # Initialize lists for fitting FBCSP
    X_all = []
    y_all = []
    
    # First Pass: Collect Data for FBCSP Fitting
    print("Collecting data for FBCSP fitting...")
    for signals_file, metadata_file in tqdm(file_pairs, desc='Collecting FBCSP Data'):
        epochs, labels = extract_features_from_file(signals_file, metadata_file)
        if epochs is not None:
            X_all.append(epochs)
            y_all.extend(labels)
    
    if not X_all:
        print("No epochs extracted from any files.")
        return
    
    X_all = np.concatenate(X_all, axis=0)  # Shape: (total_epochs, n_channels, n_times)
    y_all = np.array(y_all)
    
    # Subsample for FBCSP fitting if necessary
    MAX_EPOCHS_FOR_FBCSP = 5000  # Further reduced
    total_epochs = X_all.shape[0]
    if total_epochs > MAX_EPOCHS_FOR_FBCSP:
        print(f"Subsampling {MAX_EPOCHS_FOR_FBCSP} epochs out of {total_epochs} for FBCSP fitting.")
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(total_epochs, size=MAX_EPOCHS_FOR_FBCSP, replace=False)
        X_sample = X_all[indices]
        y_sample = y_all[indices]
    else:
        X_sample = X_all
        y_sample = y_all
    
    # Fit FBCSP on the sampled data
    print("Fitting FBCSP on sampled data...")
    try:
        fbcsp.fit(X_sample, y_sample)
    except ValueError as e:
        print(f"Error during FBCSP fitting: {e}")
        return
    
    # Clear memory
    del X_all
    del y_all
    del X_sample
    del y_sample
    
    # Second Pass: Extract Features and Save to HDF5
    print("Extracting features and saving to HDF5...")
    with h5py.File(FEATURES_HDF5_PATH, 'a') as h5f:
        feature_grp = h5f['features']
        label_grp = h5f['labels']
        participant_grp = h5f['participant_ids']
        
        idx = 0  # Dataset index
        for signals_file, metadata_file in tqdm(file_pairs, desc='Processing and Saving Features'):
            epochs, labels = extract_features_from_file(signals_file, metadata_file)
            if epochs is None:
                continue
            
            # Transform features
            try:
                X_features = fbcsp.transform(epochs)  # Shape: (n_epochs, n_features)
            except ValueError as e:
                print(f"Error during feature transformation for {signals_file}: {e}")
                continue
            
            # Feature Selection (Select Top K Features)
            selector = SelectKBest(score_func=mutual_info_classif, k=50)  # Adjust k as needed
            X_selected = selector.fit_transform(X_features, labels)
            
            # Standardize Features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            
            # Save to HDF5
            dataset_name = f"subject_{idx}"
            feature_grp.create_dataset(dataset_name, data=X_scaled, compression="gzip")
            label_grp.create_dataset(dataset_name, data=labels, compression="gzip")
            participant_id = signals_file.split('_')[0]  # Assuming 'S01' from 'S01_B_RWEO_PreOL_signals.csv'
            participant_grp.create_dataset(dataset_name, data=np.string_(participant_id), compression="gzip")
            idx += 1
    
    print("Feature extraction and saving completed.")

if __name__ == "__main__":
    main()



