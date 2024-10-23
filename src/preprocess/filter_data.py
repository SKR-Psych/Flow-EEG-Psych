import os
import mne
import pandas as pd

# Path to processed data folder containing CSV files
processed_data_path = 'data/processed/'

# Output folder for filtered data
filtered_data_path = 'data/filtered/'

# Create output directory if it doesn't exist
if not os.path.exists(filtered_data_path):
    os.makedirs(filtered_data_path)

def bandpass_filter(csv_file):
    # Load the CSV data
    df = pd.read_csv(csv_file)
    
    # Extract channel names from columns
    channel_names = df.columns.tolist()
    
    # Convert DataFrame to NumPy array for MNE processing
    data = df.to_numpy().T  # Shape should be (n_channels, n_times)

    # Create MNE RawArray
    info = mne.create_info(ch_names=channel_names, sfreq=256, ch_types='eeg')  # Assuming sfreq of 256 Hz
    raw = mne.io.RawArray(data, info)
    
    # Apply bandpass filter with parameters to accommodate short signal length
    raw.filter(l_freq=1.0, h_freq=40.0, filter_length='auto', phase='zero', fir_window='hamming', fir_design='firwin2')

    # Convert back to DataFrame
    filtered_data = raw.get_data().T
    filtered_df = pd.DataFrame(filtered_data, columns=channel_names)
    
    # Save the filtered data
    output_file = os.pat

