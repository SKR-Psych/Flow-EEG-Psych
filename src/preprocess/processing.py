import os
import numpy as np
import pandas as pd
from load_data import load_mat_file, get_mat_file_list

def convert_mat_to_csv():
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '../../data/raw/')
    processed_data_dir = os.path.join(current_dir, '../../data/processed/')

    os.makedirs(processed_data_dir, exist_ok=True)
    mat_files = get_mat_file_list(raw_data_dir)

    for mat_file in mat_files:
        print(f'Processing {mat_file}...')

        try:
            mat = load_mat_file(os.path.join(raw_data_dir, mat_file))

            # Navigate the HDF5 structure
            EEG_full = mat['BF_Data']['actualVariable']['EEG_full']

            # Extract data
            data = np.array(EEG_full['data']).T  # Transpose to get timepoints x channels

            # Extract channel labels
            chanlocs = EEG_full['chanlocs']
            channel_labels = []
            for i in range(len(chanlocs)):
                label = ''.join(chr(c[0]) for c in mat[chanlocs[i]]['labels'])
                channel_labels.append(label)
            signal_df = pd.DataFrame(data, columns=channel_labels)

            # Save signals to CSV
            signal_csv_filename = mat_file.replace('.mat', '_signals.csv')
            signal_csv_path = os.path.join(processed_data_dir, signal_csv_filename)
            signal_df.to_csv(signal_csv_path, index=False)
            print(f'Saved signals to {signal_csv_path}')

            # Process events
            events = EEG_full['event']
            event_list = []
            for i in range(len(events)):
                event = events[i]
                event_dict = {}
                for key in mat[events[i]].keys():
                    value = mat[events[i]][key][()]
                    if isinstance(value, np.ndarray) and value.size == 1:
                        value = value.item()
                    event_dict[key] = value
                event_list.append(event_dict)

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



