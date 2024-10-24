import h5py
import os

# Directory containing raw .mat files
raw_data_dir = "data/raw/"

for file in os.listdir(raw_data_dir):
    if file.endswith('.mat'):
        file_path = os.path.join(raw_data_dir, file)
        try:
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                print(f"{file}: {keys}")
                
                # Check what is inside 'actualVariable'
                if 'actualVariable' in keys:
                    actual_variable = f['actualVariable']
                    if isinstance(actual_variable, h5py.Group):
                        print(f"Contents of 'actualVariable' in {file}: {list(actual_variable.keys())}")

                        # Check if 'EEG_full' is inside 'actualVariable'
                        if 'EEG_full' in actual_variable:
                            eeg_full = actual_variable['EEG_full']
                            print(f"'EEG_full' shape in {file}: {eeg_full.shape}")
                        else:
                            print(f"'EEG_full' not found in 'actualVariable' in {file}")
                    else:
                        print(f"'actualVariable' in {file} is not a group.")
        except Exception as e:
            print(f"Error opening {file}: {e}")
