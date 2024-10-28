import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne.channels import make_standard_montage
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

# Define EEG Channels (first 64 columns based on your sample)
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
            print(f"Fitting CSP for {band_name} band: {fmin}-{fmax} Hz")
            # Bandpass filter
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
            fmin, fmax = self.filter_params[band_name]
            # Bandpass filter using stored filter params
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
        epoch_data = signals_df.iloc[start_sample:end_sample][EEG_CHANNELS].values.T  # shape: (n_channels, n_times)
        
        epochs.append(epoch_data.astype(np.float32))  # Use float32 to reduce memory
        labels.append(label)
    
    print(f"Total events: {total_events}, Skipped events: {skipped_events}, Extracted epochs: {len(epochs)}")
    return np.array(epochs), np.array(labels)

def main():
    print("Starting feature extraction...")
    
    # Initialize Feature Extractor
    fbcsp = FBCSPExtractor(frequency_bands=FREQUENCY_BANDS, n_components=N_COMPONENTS)
    
    # Initialize Lists to Store Data
    all_labels = []
    participant_ids = []
    
    # Get list of signals CSV files
    signals_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_signals.csv')]
    
    if not signals_files:
        print("No signals CSV files found in the processed data directory.")
        return
    
    batch_size = 1000  # Adjust this based on available memory
    
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
        
        # Extract epochs and labels in batches
        for start in range(0, len(signals_df), batch_size):
            end = start + batch_size
            epochs, labels = extract_epochs(signals_df.iloc[start:end], metadata_df, sfreq=SFREQ, tmin=0, tmax=2)
            
            if len(epochs) == 0:
                print(f"No valid epochs extracted from {signals_file}.")
                continue
            
            # Fit and transform each batch to avoid memory overload
            fbcsp.fit(epochs.astype(np.float64), labels)
            X_features = fbcsp.transform(epochs.astype(np.float64))
            
            all_labels.extend(labels)
            participant_ids.extend([base_name] * len(labels))
            
            # Save batch features to avoid memory overload
            batch_features_df = pd.DataFrame(X_features)
            batch_features_df['label'] = labels
            batch_features_df['participant_id'] = participant_ids




