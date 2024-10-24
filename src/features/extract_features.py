# Feature Extraction from STFT Data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from src.features.extract_features import extract_features_from_file

# Paths to STFT data and features output
data_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '..', 'data', 'stft')
features_output_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '..', 'data', 'features')

# Ensure the features directory exists
if not os.path.exists(features_output_path):
    os.makedirs(features_output_path)

# Main function to extract features from all STFT files
def main():
    feature_list = []
    try:
        # List all files in the data path to ensure there are files to process
        files = os.listdir(data_path)
        if not files:
            print("No files found in the data path. Please check the path and try again.")
            return

        # Extract features from each file
        for file in files:
            if file.endswith('_stft.csv'):
                file_path = os.path.join(data_path, file)
                print(f"Processing file: {file}")
                features = extract_features_from_file(file_path)

                # Debugging: Print the extracted features for verification
                if features:
                    print(f"Extracted Features for {file}: {features}")
                else:
                    print(f"Warning: No features extracted for {file}. Please check the data.")

                # Extract subject and condition from filename
                filename_parts = file.split('_')
                subject = filename_parts[0]
                condition = '_'.join(filename_parts[1:-1])

                # Add subject and condition to features if they exist
                if features:
                    features['subject'] = subject
                    features['condition'] = condition
                    feature_list.append(features)
    except FileNotFoundError:
        print(f"The directory {data_path} does not exist. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Save the extracted features if available
    if feature_list:
        features_df = pd.DataFrame(feature_list)
        features_output_file = os.path.join(features_output_path, 'features.csv')
        features_df.to_csv(features_output_file, index=False)
        print(f"Feature extraction complete. Features saved to: {features_output_file}")
    else:
        print("No features extracted. Please check the input files and try again.")

if __name__ == "__main__":
    main()
