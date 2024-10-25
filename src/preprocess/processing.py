import os
import numpy as np
import pandas as pd
from load_data import load_mat_file, get_mat_file_list

def convert_mat_to_csv():
    # Define paths relative to this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '../../data/raw/')
    processed_data_dir = os.path.join(current_dir, '../../data/processed/')

    # Ensure the processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # Get list of .mat files
    mat_files = get_mat_file_list(raw_data_dir)

    for mat_file in mat_files:
        print(f'Processing {mat_file}...')

        # Load the .mat file
        mat = load_mat_file(os.path.join(raw_data_dir, mat_file))

        # Extract the EEG data structure
        try:
            BF_Data = mat['BF_Data']
            EEG_full = BF_Data.actualVariable.EEG_full

            # Extract data
            data = EEG_full.data  # Shape: (channels x timepoints)
            data = data.astype(np.float64)  # Ensure data is in float64 format

            # Extract event markers
            events = EEG_full.event  # List of events

            # Extract channel information
            chanlocs = EEG_full.chanlocs  # List of channel info

            # Create a DataFrame for the signals
            data = data.T  # Transpose to get timepoints x channels

            # Extract channel labels
            channel_labels = []
            for chan in np.ravel(chanlocs):
                label = chan.labels
                channel_labels.append(label)
            signal_df = pd.DataFrame(data, columns=channel_labels)

            # Save signals to CSV
            signal_csv_filename = mat_file.replace('.mat', '_signals.csv')
            signal_csv_path = os.path.join(processed_data_dir, signal_csv_filename)
            signal_df.to_csv(signal_csv_path, index=False)
            print(f'Saved signals to {signal_csv_path}')

            # Process events
            event_list = []
            for event in np.ravel(events):
                event_dict = {}
                for field_name in event._fieldnames:
                    value = getattr(event, field_name)
                    event_dict[field_name] = value
                event_list.append(event_dict)

            # Create a DataFrame for the metadata/events
            metadata_df = pd.DataFrame(event_list)

            # Save metadata to CSV
            metadata_csv_filename = mat_file.replace('.mat', '_metadata.csv')
            metadata_csv_path = os.path.join(processed_data_dir, metadata_csv_filename)
            metadata_df.to_csv(metadata_csv_path, index=False)
            print(f'Saved metadata to {metadata_csv_path}')

        except Exception as e:
            print(f'Error processing {mat_file}: {e}')
            continue

    print('All files processed.')

if __name__ == '__main__':
    convert_mat_to_csv()


