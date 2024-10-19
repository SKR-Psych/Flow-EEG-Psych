import os
import h5py
import pandas as pd

# Define path to raw data folder
raw_data_path = 'data/raw/'

def load_mat_file(filename):
    filepath = os.path.join(raw_data_path, filename)
    
    # Extract subject and condition from the filename based on naming convention
    file_parts = filename.split('_')
    subject = file_parts[0]  # e.g., "S01"
    condition = '_'.join(file_parts[1:])[:-4]  # Everything after subject, removing '.mat'

    with h5py.File(filepath, 'r') as mat_data:
        # Print keys to understand the structure of the file
        print("Keys in the file:", list(mat_data.keys()))
        
        # Explore 'actualVariable' content
        if 'actualVariable' in mat_data:
            actual_variable = mat_data['actualVariable']
            
            # Check if 'actualVariable' is a group
            if isinstance(actual_variable, h5py.Group):
                print("'actualVariable' is a group. Keys in the group:", list(actual_variable.keys()))

                # Explore 'EEG_full' within 'actualVariable'
                if 'EEG_full' in actual_variable:
                    eeg_full = actual_variable['EEG_full']

                    # Check if 'EEG_full' is a group
                    if isinstance(eeg_full, h5py.Group):
                        print("'EEG_full' is a group. Keys in the group:", list(eeg_full.keys()))

                        # Extract relevant data from 'EEG_full'
                        eeg_data = eeg_full['data'][:]
                        sampling_rate = eeg_full['srate'][()]

                        # Print extracted information for verification
                        print(f"EEG data shape: {eeg_data.shape}")
                        print(f"Sampling Rate: {sampling_rate}")
                        print(f"Subject (from filename): {subject}")
                        print(f"Condition (from filename): {condition}")

                        # Convert EEG data to DataFrame for easier manipulation
                        num_channels = eeg_data.shape[0]
                        num_samples = eeg_data.shape[1]
                        
                        # Extract channel names (assuming it's a dataset of strings)
                        chanlocs = []
                        if 'chanlocs' in eeg_full:
                            chanlocs_data = eeg_full['chanlocs']
                            for i in range(num_channels):
                                try:
                                    chanloc = chanlocs_data[i][()].tobytes().decode('utf-8').strip()
                                    chanlocs.append(chanloc)
                                except:
                                    chanlocs.append(f"Channel_{i+1}")
                        else:
                            chanlocs = [f"Channel_{i+1}" for i in range(num_channels)]

                        df_signal = pd.DataFrame(eeg_data.T, columns=chanlocs)

                        # Print first few rows of the signal DataFrame
                        print(df_signal.head())

                        # Metadata dictionary with values extracted from filename
                        metadata = {
                            'subject': subject,
                            'condition': condition,
                            'sampling_rate': sampling_rate
                        }

                        # Return the signal data and metadata
                        return df_signal, metadata
    return None, None

# Save Data Function
def save_data(df_signal, metadata, output_dir='data/processed/'):
    # Create processed directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Handle metadata fields to avoid null characters
    subject_id = metadata.get('subject', 'unknown').replace('\x00', '').strip() or "Unknown"
    condition = metadata.get('condition', 'unknown').replace('\x00', '').strip() or "Unknown"

    # Save the EEG data
    signal_filename = f"{subject_id}_{condition}_signals.csv"
    df_signal.to_csv(os.path.join(output_dir, signal_filename), index=False)
    
    # Save metadata
    metadata_filename = f"{subject_id}_{condition}_metadata.csv"
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(os.path.join(output_dir, metadata_filename), index=False)

# Example usage:
def main():
    for file in os.listdir(raw_data_path):
        if file.endswith('.mat'):
            df_signal, metadata = load_mat_file(file)
            if df_signal is not None:
                save_data(df_signal, metadata)
                print(f"Saved data for subject {metadata['subject']} under condition {metadata['condition']}")

if __name__ == "__main__":
    main()



