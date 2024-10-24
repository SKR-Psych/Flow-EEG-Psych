import h5py
import os

# Directory containing raw .mat files
raw_data_dir = "data/raw/"

def inspect_group(group, indent=0):
    """Recursively prints the contents of an HDF5 group."""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{' ' * indent}Group: {key}")
            inspect_group(item, indent + 2)
        elif isinstance(item, h5py.Dataset):
            print(f"{' ' * indent}Dataset: {key}, Shape: {item.shape}")

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

                        # Check the 'EEG_full' group
                        if 'EEG_full' in actual_variable:
                            eeg_full = actual_variable['EEG_full']
                            if isinstance(eeg_full, h5py.Group):
                                print(f"'EEG_full' is a group in {file}, exploring contents...")
                                inspect_group(eeg_full)
                            elif isinstance(eeg_full, h5py.Dataset):
                                print(f"'EEG_full' is a dataset in {file}, shape: {eeg_full.shape}")
                        else:
                            print(f"'EEG_full' not found in 'actualVariable' in {file}")
                    else:
                        print(f"'actualVariable' in {file} is not a group.")
        except Exception as e:
            print(f"Error opening {file}: {e}")

