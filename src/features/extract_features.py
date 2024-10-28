# src/features/extract_features.py

import os
import numpy as np
import pandas as pd
import scipy.io
import mne
from mne.preprocessing import ICA, create_eog_epochs
from mne.channels import make_standard_montage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from tqdm import tqdm

# Define Constants
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW_EEG_DIR = os.path.join(BASE_DIR, 'data', 'raw')
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'features')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'decoders')
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

    def fit(self, X, y):
        for band_name, (fmin, fmax) in self.frequency_bands.items():
            # Bandpass filter
            csp = mne.decoding.CSP(n_components=self.n_components, 
                                   reg=None, 
                                   log=True, 
                                   norm_trace=False)
            # Fit CSP
            csp.fit(X, y)
            self.csp_pipelines[band_name] = csp
        return self

    def transform(self, X):
        features = []
        for band_name, csp in self.csp_pipelines.items():
            # Apply CSP
            csp_features = csp.transform(X)
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

def extract_epochs(raw, eeg_data, labels, tmin=0, tmax=2):
    """
    Extract 2-second epochs around event onsets.
    """
    # Create MNE Raw object from eeg_data
    # Assuming eeg_data is a numpy array of shape (n_channels, n_times)
    info = raw.info.copy()
    new_raw = mne.io.RawArray(eeg_data, info)
    
    # Create events array
    # Assuming labels indicate the start of each epoch
    # For example, labels = [1, 2, 1, 2, ...] alternating between conditions
    n_epochs = len(labels)
    events = []
    for i in range(n_epochs):
        events.append([i * int(tmax * raw.info['sfreq']), 0, 1])  # Dummy event ID=1
    events = np.array(events)
    
    # Extract epochs
    epochs = mne.Epochs(new_raw, events=events, event_id=1, tmin=tmin, tmax=tmax, 
                        baseline=None, preload=True, verbose=False)
    X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y = np.array(labels)
    return X, y

def main():
    # Verify RAW_EEG_DIR exists
    if not os.path.exists(RAW_EEG_DIR):
        raise FileNotFoundError(f"Raw EEG directory not found: {RAW_EEG_DIR}")
    
    # Initialize Feature Extractor
    fbcsp = FBCSPExtractor(frequency_bands=FREQUENCY_BANDS, n_components=N_COMPONENTS)
    
    # Initialize Lists to Store Data
    all_features = []
    all_labels = []
    participant_ids = []
    
    # Iterate Over Each Participant's Raw EEG File
    raw_files = [f for f in os.listdir(RAW_EEG_DIR) if f.lower().endswith('.mat')]
    
    if not raw_files:
        raise FileNotFoundError(f"No .mat files found in {RAW_EEG_DIR}")
    
    for raw_file in tqdm(raw_files, desc='Processing EEG Files'):
        participant_id = os.path.splitext(raw_file)[0]  # Filename without extension
        raw_path = os.path.join(RAW_EEG_DIR, raw_file)
        
        try:
            # Load Raw EEG Data from .mat file
            mat = scipy.io.loadmat(raw_path)
            # Replace 'EEG' and 'labels' with actual variable names in your .mat files
            # For example, if your EEG data is under the key 'EEG', use:
            eeg_data = mat['EEG']  # Shape: (n_channels, n_times)
            labels = mat['labels'].flatten()  # Assuming labels are stored in 'labels' variable
            # If labels are stored differently, adjust accordingly
        except Exception as e:
            print(f"Error loading {raw_file}: {e}")
            continue
        
        # Create MNE Raw object from eeg_data
        # Assuming eeg_data is already in the shape (n_channels, n_times)
        # and channels are ordered as per 'standard_1020' montage
        try:
            # Create info structure
            sfreq = 256  # Replace with your actual sampling frequency
            n_channels = eeg_data.shape[0]
            ch_names = [f'EEG{i}' for i in range(1, n_channels + 1)]
            ch_types = ['eeg'] * n_channels
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            raw = mne.io.RawArray(eeg_data, info, verbose=False)
        except Exception as e:
            print(f"Error creating Raw object for {raw_file}: {e}")
            continue
        
        # Preprocess EEG Data
        raw_clean = preprocess_eeg(raw)
        
        # Extract Epochs
        X, y = extract_epochs(raw_clean, eeg_data, labels, tmin=0, tmax=2)
        
        # Append to All Data
        all_features.append(X)
        all_labels.append(y)
        participant_ids.extend([participant_id]*len(y))
    
    if not all_features:
        print("No features extracted. Please check your .mat files and labels.")
        return
    
    # Concatenate All Data
    X_all = np.concatenate(all_features, axis=0)  # Shape: (total_epochs, n_channels, n_times)
    y_all = np.concatenate(all_labels, axis=0)    # Shape: (total_epochs,)
    participant_ids = np.array(participant_ids)
    
    # Shuffle Data
    X_all, y_all, participant_ids = shuffle(X_all, y_all, participant_ids, random_state=42)
    
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
    
    # Save Features and Labels
    features_df = pd.DataFrame(X_scaled)
    features_df['label'] = y_all
    features_df['participant_id'] = participant_ids
    features_df.to_csv(os.path.join(FEATURES_DIR, 'all_features.csv'), index=False)
    
    print("Feature extraction completed and saved successfully.")

if __name__ == '__main__':
    main()

