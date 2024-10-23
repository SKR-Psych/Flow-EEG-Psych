import os
import numpy as np
import pandas as pd
from scipy.signal import stft

# Path to processed data folder containing CSV files
processed_data_path = 'data/processed/'

# Output folder for STFT data
stft_data_path = 'data/stft/'

# Create output directory if it doesn't exist
if not os.path.exists(stft_data_path):
    os.makedirs(stft_data_path)

def stft_transform(csv_file, fs=256, nperseg=64):
    # Load the CSV data
    df = pd.read_csv(csv_file)
    
    # Extract channel names from columns
    channel_names = df.columns.tolist()
    
    # Convert DataFrame to NumPy array for STFT processing
    data = df.to_numpy()  # Shape: (n_samples, n_channels)

    # Initialize lists to store STFT results
    frequencies = None
    times = None
    stft_results = []

    # Perform STFT for each channel
    for i, channel in enumerate(channel_names):
        f, t, Zxx = stft(data[:, i], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        
        # Store frequency and time data once
        if frequencies is None:
            frequencies = f
            times = t

        # Store the magnitude of the STFT
        stft_results.append(np.abs(Zxx))

    # Make sure the STFT results are consistent for all channels
    num_frequencies = len(frequencies)
    num_times = len(times)

    # Flatten the STFT results for each channel and create DataFrame
    flattened_results = {}
    flattened_results['Frequency (Hz)'] = np.tile(frequencies, num_times)
    flattened_results['Time (s)'] = np.repeat(times, num_frequencies)

    for i, channel in enumerate(channel_names):
        flattened_results[channel] = stft_results[i].flatten()

    # Create a DataFrame and save the results
    output_file = os.path.join(stft_data_path, os.path.basename(csv_file).replace('_signals.csv', '_stft.csv'))
    stft_df = pd.DataFrame(flattened_results)
    stft_df.to_csv(output_file, index=False)
    print(f"STFT data saved to: {output_file}")

def main():
    # Loop through all processed CSV files and apply STFT
    print(f"Looking for files in: {processed_data_path}")
    files_found = False
    for file in os.listdir(processed_data_path):
        if file.endswith('.csv') and 'signals' in file:
            # Check if the corresponding STFT file already exists
            output_filename = file.replace('_signals.csv', '_stft.csv')
            output_filepath = os.path.join(stft_data_path, output_filename)

            if os.path.exists(output_filepath):
                print(f"Skipping already processed file: {file}")
                continue
            
            # Proceed with processing the file if not already done
            files_found = True
            file_path = os.path.join(processed_data_path, file)
            print(f"Processing file: {file_path}")
            try:
                stft_transform(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if not files_found:
        print("No files found in the processed data directory to apply STFT.")

if __name__ == "__main__":
    main()

