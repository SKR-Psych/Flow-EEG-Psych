# src/models/feature_extract.py

import os
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
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
RAW_EEG_DIR = '../../data/raw_eeg/'  # Directory containing raw EEG files
FEATURES_DIR = '../../data/features/'
MODEL_DIR = '../../models/decoders/'
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
    raw.filter(0.5, 50., fir_design='firwin')
    
    # Apply ICA for artifact removal
    ica = ICA(n_components=15, random_state=97, max_iter='auto')
    ica.fit(raw)
    
    # Find and exclude EOG artifacts
    eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
    ica.exclude = eog_indices
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    
    return raw_clean

def extract_epochs(raw, event_onsets, event_labels, tmin=0, tmax=2):
    """
    Extract 2-second epochs around event onsets.
    """
    epochs = mne.Epochs(raw, events=event_onsets, event_id=None, tmin=tmin, tmax=tmax, 
                        baseline=None, preload=True)
    X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y = np.array(event_labels)
    return X, y

def main():
    # Initialize Feature Extractor
    fbcsp = FBCSPExtractor(frequency_bands=FREQUENCY_BANDS, n_components=N_COMPONENTS)
    
    # Initialize Lists to Store Data
    all_features = []
    all_labels = []
    participant_ids = []
    
    # Iterate Over Each Participant's Raw EEG File
    raw_files = [f for f in os.listdir(RAW_EEG_DIR) if f.endswith('.edf')]
    
    for raw_file in tqdm(raw_files, desc='Processing EEG Files'):
        participant_id = raw_file.split('.')[0]  # Assuming filename starts with participant ID
        raw_path = os.path.join(RAW_EEG_DIR, raw_file)
        
        # Load Raw EEG Data
        raw = read_raw_edf(raw_path, preload=True, verbose=False)
        
        # Preprocess EEG Data
        raw_clean = preprocess_eeg(raw)
        
        # Define Event Onsets and Labels
        # Assuming you have a way to define events based on task difficulty
        # For illustration, let's create synthetic events:
        # Event ID 1: Easy (Class 1), Event ID 2: Hard (Class 2)
        # In practice, replace this with actual event markers from your data
        
        # Example: Extracting based on time segments
        # Let's assume that every 2 seconds is an epoch
        n_samples = raw_clean.n_times
        sfreq = raw_clean.info['sfreq']
        epoch_length = 2  # seconds
        samples_per_epoch = int(epoch_length * sfreq)
        
        # Generate synthetic event onsets and labels
        # Replace this with actual event extraction logic
        event_onsets = []
        event_labels = []
        for i in range(0, n_samples, samples_per_epoch):
            if i + samples_per_epoch > n_samples:
                break
            event_onsets.append([0, 0, 1])  # Dummy event
            # Alternate labels for easy and hard
            label = 1 if (i // samples_per_epoch) % 2 == 0 else 2
            event_labels.append(label)
        
        # Extract Epochs
        X, y = extract_epochs(raw_clean, event_onsets, event_labels, tmin=0, tmax=2)
        
        # Reshape X for FBCSP: (n_epochs, n_channels, n_times)
        # FBCSP expects (n_epochs, n_channels, n_times)
        
        # Append to All Data
        all_features.append(X)
        all_labels.append(y)
        participant_ids.extend([participant_id]*len(y))
    
    # Concatenate All Data
    X_all = np.concatenate(all_features, axis=0)  # Shape: (total_epochs, n_channels, n_times)
    y_all = np.concatenate(all_labels, axis=0)    # Shape: (total_epochs,)
    participant_ids = np.array(participant_ids)
    
    # Shuffle Data
    X_all, y_all, participant_ids = shuffle(X_all, y_all, participant_ids, random_state=42)
    
    # Flatten X for FBCSP: MNE expects (n_epochs, n_channels, n_times)
    # Already in correct shape
    
    # Fit FBCSP
    fbcsp.fit(X_all, y_all)
    X_features = fbcsp.transform(X_all)  # Shape: (n_epochs, n_features)
    
    # Standardise Features
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
    import mne  # Ensure MNE is imported here
    main()
