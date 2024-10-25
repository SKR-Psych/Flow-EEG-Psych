# src/models/train_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(current_dir, '../../data/features/')
    features_file = os.path.join(features_dir, 'all_features.csv')

    # Load features
    df = pd.read_csv(features_file)
    print('Features loaded successfully.')

    # Preprocessing
    # Handle missing values
    df = df.dropna()
    print('Missing values handled.')

    # Encode labels
    label_encoder = LabelEncoder()
    df['event_type_encoded'] = label_encoder.fit_transform(df['event_type'])
    num_classes = len(label_encoder.classes_)
    print('Labels encoded.')

    # Sort by participant_id and timestamp to maintain temporal order
    df = df.sort_values(by=['participant_id', 'timestamp'])
    print('Data sorted by participant and timestamp.')

    # Feature columns
    feature_columns = [
        'mean', 'std', 'var', 'skew', 'kurtosis', 'zero_crossing_rate',
        'bandpower_delta', 'relative_power_delta', 'bandpower_theta',
        'relative_power_theta', 'bandpower_alpha', 'relative_power_alpha',
        'bandpower_beta', 'relative_power_beta', 'bandpower_gamma',
        'relative_power_gamma'
    ]

    # Scale features
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    print('Features scaled.')

    # Generate sequences for each participant and channel
    sequence_length = 5  # Adjust as needed
    sequences = []
    labels = []
    groups = []  # To keep track of participant IDs
    participants = df['participant_id'].unique()

    for participant in participants:
        participant_df = df[df['participant_id'] == participant]
        channels = participant_df['channel'].unique()

        for channel in channels:
            channel_df = participant_df[participant_df['channel'] == channel]
            channel_df = channel_df.reset_index(drop=True)
            data = channel_df[feature_columns].values
            label = channel_df['event_type_encoded'].values

            # Create sequences
            for i in range(len(data) - sequence_length + 1):
                seq = data[i:i+sequence_length]
                lbl = label[i+sequence_length-1]  # Label for the sequence
                sequences.append(seq)
                labels.append(lbl)
                groups.append(participant)

    sequences = np.array(sequences)
    labels = np.array(labels)
    print(f'Total sequences generated: {sequences.shape[0]}')

    # Convert labels to categorical
    labels_categorical = to_categorical(labels, num_classes=num_classes)

    # Split data while avoiding data leakage (GroupShuffleSplit by participant)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(sequences, labels, groups=groups))

    X_train, X_test = sequences[train_idx], sequences[test_idx]
    y_train, y_test = labels_categorical[train_idx], labels_categorical[test_idx]
    print('Data split into training and testing sets.')

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, len(feature_columns)), return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model compiled.')

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    print('Model trained.')

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.2f}')

    # Save the model
    model_save_path = os.path.join(current_dir, 'trained_lstm_model.h5')
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == '__main__':
    main()
