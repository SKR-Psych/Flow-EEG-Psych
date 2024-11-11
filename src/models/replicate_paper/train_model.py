# models/replicate_paper/train_model.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tqdm import tqdm
import warnings
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'replicate_paper')
os.makedirs(MODEL_DIR, exist_ok=True)

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

def load_data():
    X = []
    y = []
    participant_ids = []
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"Processed data directory does not exist: {PROCESSED_DATA_DIR}")
        return None, None, None
    
    processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_signals.csv')]
    
    print(f"Looking for processed files in: {PROCESSED_DATA_DIR}")
    print(f"Files found: {processed_files}")
    
    if not processed_files:
        print("No '_signals.csv' files found in the processed data directory.")
        return None, None, None
    
    for signals_file in tqdm(processed_files, desc='Loading Data'):
        base_name = signals_file.replace('_signals.csv', '')
        metadata_file = f"{base_name}_metadata.csv"
        metadata_path = os.path.join(PROCESSED_DATA_DIR, metadata_file)
        signals_path = os.path.join(PROCESSED_DATA_DIR, signals_file)
        
        if not os.path.exists(metadata_path):
            print(f"Metadata file {metadata_file} not found for signals file {signals_file}. Skipping.")
            continue
        
        try:
            signals_df = pd.read_csv(signals_path)
            if not set(EEG_CHANNELS).issubset(signals_df.columns):
                print(f"Signals file {signals_file} is missing some EEG channels. Skipping.")
                continue
            signals_df = signals_df[EEG_CHANNELS].astype(np.float32)
            
            metadata_df = pd.read_csv(metadata_path)
            epochs, labels = extract_epochs(signals_df, metadata_df, sfreq=256, tmin=0, tmax=2)
            
            if epochs is not None and len(epochs) > 0:
                X.append(epochs)
                y.extend(labels)
                participant_ids.extend([base_name] * len(labels))
        except Exception as e:
            print(f"Error processing {signals_file}: {e}")
            traceback.print_exc()
            continue
    
    if X:
        X = np.concatenate(X, axis=0)
        y = np.array(y)
        participant_ids = np.array(participant_ids)
        return X, y, participant_ids
    else:
        return None, None, None

def extract_epochs(signals_df, metadata_df, sfreq=256, tmin=0, tmax=2):
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
        epoch_data = signals_df.iloc[start_sample:end_sample].values.flatten()
        
        epochs.append(epoch_data)
        labels.append(label)
    
    print(f"Total events: {total_events}, Skipped events: {skipped_events}, Extracted epochs: {len(epochs)}")
    return np.array(epochs), np.array(labels)

def main():
    print("Loading data...")
    X, y, participant_ids = load_data()
    
    if X is None:
        print("No data loaded. Exiting.")
        return
    
    print(f"Total samples: {X.shape[0]}")
    print(f"Feature size: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    print("Training the model...")
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()
        return
    
    # Evaluate the model
    print("Evaluating the model...")
    try:
        y_pred = clf.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        traceback.print_exc()
        return
    
    # Save the model and scaler
    try:
        model_path = os.path.join(MODEL_DIR, 'random_forest_model.joblib')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    except Exception as e:
        print(f"Error saving model or scaler: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()


