# src/preprocess/processing.py

import os
import numpy as np
import pandas as pd
from load_data import load_mat_file, get_mat_file_list

def convert_mat_to_csv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '../../data/raw/')
    processed_data_dir = os.path.join(current_dir, '../../data/processed/')

    os.makedirs(processed_data_dir, exist_ok=True)
    mat_files = get_mat_file_list(raw_data_dir)

    for mat_file in mat_files:
        print(f'Processing {mat_file}...')

        try:
            mat = load_mat_file(os.path.join(raw_data_dir, mat_file))

            # Access 'actualVariable'
            actual_variable = mat.get('actualVariable', None)
            if actual_variable is None:
                print(f"Error: {mat_file} does not contain 'actualVariable'.")
                continue

            # Access 'EEG_full' within 'actualVariable'
            EEG_full = actual_variable.get('EEG_full', None)
            if EEG_full is None:
                print(f"Error: {mat_file} does not contain 'EEG_full' within 'actualVariable'.")
                continue

            # Extract data
            data = EEG_full.get('data', None)
            if data is None:
                print(f"Error: 'EEG_full' does not contain 'data' in {mat_file}.")
                continue

            data = np.array(data).astype(np.float64)

            # Extract channel labels
            chanlocs = EEG_full.get('chanlocs', [])
            channel_labels = []
            for chan in chanlocs:
                label = chan.get('labels', 'Unknown')
                channel_labels.append(label)

            # Create DataFrame and save signals
            data = data.T  # Transpose to get timepoints x channels
            signal_df = pd.DataFrame(data, columns=channel_labels)
            signal_csv_filename = mat_file.replace('.mat', '_signals.csv')
            signal_csv_path = os.path.join(processed_data_dir, signal_csv_filename)
            signal_df.to_csv(signal_csv_path, index=False)
            print(f'Saved signals to {signal_csv_path}')

            # Extract events
            events = EEG_full.get('event', [])
            event_list = []
            for event in events:
                event_dict = {}
                for key, value in event.items():
                    # Handle nested arrays or structures if necessary
                    if isinstance(value, (np.ndarray, list)) and len(value) == 1:
                        value = value[0]
                    event_dict[key] = value
                event_list.append(event_dict)

            # Create DataFrame and save metadata
            if event_list:
                metadata_df = pd.DataFrame(event_list)
                metadata_csv_filename = mat_file.replace('.mat', '_metadata.csv')
                metadata_csv_path = os.path.join(processed_data_dir, metadata_csv_filename)
                metadata_df.to_csv(metadata_csv_path, index=False)
                print(f'Saved metadata to {metadata_csv_path}')
            else:
                print(f"No events found in {mat_file}.")

        except Exception as e:
            print(f'Error processing {mat_file}: {e}')
            continue

    print('All files processed.')

if __name__ == '__main__':
    convert_mat_to_csv()




