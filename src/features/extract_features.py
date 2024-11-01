# src/features/extract_features.py

import os
import numpy as np
import pandas as pd
import mne
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tqdm import tqdm
import h5py
import warnings
import traceback

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
            try:
                X_filtered = mne.filter.filter_data(X, self.sfreq, l_freq=fmin, h_freq=fmax, verbose=False)
            except Exception as e:
                print(f"Filtering failed for band {band_name}: {e}")
                raise
            
            # Initialize and fit CSP
            try:
                csp = mne.decoding.CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
                csp.fit(X_filtered, y)
            except Exception as e:
                print(f"CSP fitting failed for band {band_name}: {e}")
                raise
            
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
            try:
                X_filtered = mne.filter.filter_data(X, self.sfreq, l_freq=fmin, h_freq=fmax, verbose=False)
            except Exception as e:
                print(f"Filtering failed during transform for band {band_name}: {e}")
                raise
            # Apply CSP
            try:
                csp_features = csp.transform(X_filtered)
            except Exception as e:
                print(f"CSP transformation failed for band {band_name}: {e}")
                raise
            features.append(csp_features)
        # Concatenate features from all bands
        if features:
            return np.hstack(features)
        else:
            return np.array([])

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
        traceback.print_exc()
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
    
    # Initialize FBCSP Extractor with a small subset for fitting
    fbcsp = FBCSPExtractor(frequency_bands=FREQUENCY_BANDS, n_components=N_COMPONENTS, sfreq=SFREQ)
    
    # Prepare HDF5 file for storing features
    try:
        h5f = h5py.File(FEATURES_HDF5_PATH, 'w')
        h5f.create_group('features')
        h5f.create_group('labels')
        h5f.create_group('participant_ids')
    except Exception as e:
        print(f"Failed to create HDF5 file: {e}")
        return
    
    # Get list of processed files
    processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_signals.csv')]
    
    if not processed_files:
        print("No processed signal files found.")
        h5f.close()
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
        h5f.close()
        return
    
    # Initialize lists for fitting FBCSP
    X_all = []
    y_all = []
    
    # First Pass: Collect a Small Subset for FBCSP Fitting
    print("Collecting a subset of data for FBCSP fitting...")
    subset_size = 500  # Number of epochs to collect
    collected = 0
    for signals_file, metadata_file in tqdm(file_pairs, desc='Collecting Subset Data', total=len(file_pairs)):
        epochs, labels = extract_features_from_file(signals_file, metadata_file)
        if epochs is not None and len(epochs) > 0:
            X_all.append(epochs)
            y_all.extend(labels)
            collected += len(epochs)
            if collected >= subset_size:
                break
    
    if not X_all:
        print("No epochs extracted from any files for FBCSP fitting.")
        h5f.close()
        return
    
    X_all = np.concatenate(X_all, axis=0)  # Shape: (subset_epochs, n_channels, n_times)
    y_all = np.array(y_all)
    
    # If collected more than subset_size, truncate
    if X_all.shape[0] > subset_size:
        X_all = X_all[:subset_size]
        y_all = y_all[:subset_size]
    
    # Fit FBCSP on the subset data
    print("Fitting FBCSP on subset data...")
    try:
        fbcsp.fit(X_all, y_all)
    except ValueError as e:
        print(f"Error during FBCSP fitting: {e}")
        traceback.print_exc()
        h5f.close()
        return
    except Exception as e:
        print(f"Unexpected error during FBCSP fitting: {e}")
        traceback.print_exc()
        h5f.close()
        return
    
    # Clear memory
    del X_all
    del y_all
    
    # Second Pass: Extract Features and Save to HDF5
    print("Extracting features and saving to HDF5...")
    feature_grp = h5f['features']
    label_grp = h5f['labels']
    participant_grp = h5f['participant_ids']
    
    idx = 0  # Dataset index
    for signals_file, metadata_file in tqdm(file_pairs, desc='Processing and Saving Features', total=len(file_pairs)):
        epochs, labels = extract_features_from_file(signals_file, metadata_file)
        if epochs is None or len(epochs) == 0:
            print(f"No valid epochs for {signals_file}. Skipping.")
            continue
        
        # Transform features
        try:
            X_features = fbcsp.transform(epochs)  # Shape: (n_epochs, n_features)
        except ValueError as e:
            print(f"Error during feature transformation for {signals_file}: {e}")
            traceback.print_exc()
            continue
        except Exception as e:
            print(f"Unexpected error during feature transformation for {signals_file}: {e}")
            traceback.print_exc()
            continue
        
        if X_features.size == 0:
            print(f"Transformed features are empty for {signals_file}. Skipping.")
            continue
        
        # Check if there are at least two unique labels
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            print(f"Not enough unique classes in labels for {signals_file}. Skipping.")
            continue
        
        # Feature Selection (Select Top K Features)
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=50)  # Adjust k as needed
            X_selected = selector.fit_transform(X_features, labels)
        except ValueError as e:
            print(f"Error during feature selection for {signals_file}: {e}")
            traceback.print_exc()
            continue
        except Exception as e:
            print(f"Unexpected error during feature selection for {signals_file}: {e}")
            traceback.print_exc()
            continue
        
        if X_selected.size == 0:
            print(f"Selected features are empty for {signals_file}. Skipping.")
            continue
        
        # Standardize Features
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
        except ValueError as e:
            print(f"Error during feature scaling for {signals_file}: {e}")
            traceback.print_exc()
            continue
        except Exception as e:
            print(f"Unexpected error during feature scaling for {signals_file}: {e}")
            traceback.print_exc()
            continue
        
        # Save to HDF5
        try:
            dataset_name = f"subject_{idx}"
            feature_grp.create_dataset(dataset_name, data=X_scaled, compression="gzip")
            label_grp.create_dataset(dataset_name, data=labels, compression="gzip")
            participant_id = signals_file.split('_')[0]  # Assuming 'S01' from 'S01_B_RWEO_PreOL_signals.csv'
            participant_grp.create_dataset(dataset_name, data=np.string_(participant_id), compression="gzip")
            idx += 1
        except Exception as e:
            print(f"Error saving data for {signals_file}: {e}")
            traceback.print_exc()
            continue
    
    h5f.close()
    print("Feature extraction and saving completed.")

if __name__ == "__main__":
    main()



