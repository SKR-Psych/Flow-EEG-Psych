import os
import h5py

# Define path to raw data folder
raw_data_path = 'data/raw/'

def load_mat_file(filename):
    filepath = os.path.join(raw_data_path, filename)
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

                    # Check if 'EEG_full' is a dataset
                    if isinstance(eeg_full, h5py.Dataset):
                        print(f"'EEG_full' is a dataset with shape {eeg_full.shape}")
                        print("First few values of 'EEG_full':", eeg_full[:5])  # Adjust as needed to explore more
                    elif isinstance(eeg_full, h5py.Group):
                        print("'EEG_full' is a group. Keys in the group:", list(eeg_full.keys()))

# Example usage:
def main():
    # Load a sample .mat file to understand its structure
    for file in os.listdir(raw_data_path):
        if file.endswith('.mat'):
            load_mat_file(file)
            break  # Load and print only the first .mat file for inspection

if __name__ == "__main__":
    main()



