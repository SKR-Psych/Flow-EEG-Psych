# src/models/evaluate_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import LSTMModel
from dataset import EEGDataset
from utils import set_seed

def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(current_dir, '../../data/features/')
    features_file = os.path.join(features_dir, 'all_features.csv')
    model_save_path = os.path.join(current_dir, 'trained_lstm_model.pth')
    log_dir = os.path.join(current_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Load features
    df = pd.read_csv(features_file)
    print('Features loaded successfully.')

    # Preprocessing
    df = df.dropna()
    print('Missing values handled.')

    label_encoder = LabelEncoder()
    df['event_type_encoded'] = label_encoder.fit_transform(df['event_type'])
    num_classes = len(label_encoder.classes_)
    print(f'Labels encoded. Classes: {label_encoder.classes_}')

    df = df.sort_values(by=['participant_id', 'timestamp'])
    print('Data sorted by participant and timestamp.')

    feature_columns = [
        'mean', 'std', 'var', 'skew', 'kurtosis', 'zero_crossing_rate',
        'bandpower_delta', 'relative_power_delta', 'bandpower_theta',
        'relative_power_theta', 'bandpower_alpha', 'relative_power_alpha',
        'bandpower_beta', 'relative_power_beta', 'bandpower_gamma',
        'relative_power_gamma'
    ]

    # Split data before scaling to prevent data leakage
    sequence_length = 5
    sequences = []
    labels = []
    groups = []
    participants = df['participant_id'].unique()

    print('Generating sequences...')
    for participant in tqdm(participants, desc='Participants'):
        participant_df = df[df['participant_id'] == participant]
        channels = participant_df['channel'].unique()

        for channel in channels:
            channel_df = participant_df[participant_df['channel'] == channel]
            channel_df = channel_df.reset_index(drop=True)
            data = channel_df[feature_columns].values
            label = channel_df['event_type_encoded'].values

            for i in range(len(data) - sequence_length + 1):
                seq = data[i:i+sequence_length]
                lbl = label[i+sequence_length-1]
                sequences.append(seq)
                labels.append(lbl)
                groups.append(participant)

    sequences = np.array(sequences)
    labels = np.array(labels)
    print(f'Total sequences generated: {sequences.shape[0]}')

    # Split data using GroupShuffleSplit to ensure no leakage between participants
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(sequences, labels, groups=groups))

    X_train, X_test = sequences[train_idx], sequences[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(f'Data split into training and testing sets.')
    print(f'Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}')

    # Scaling: Fit on training data only
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    print('Features scaled.')

    # Create Datasets and DataLoaders
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)  # Increased batch size

    # Initialize the model
    input_size = sequences.shape[2]
    hidden_size = 128  # Must match training
    num_layers = 3      # Must match training
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout=0.5).to(device)
    print('Model initialized.')

    # Load the trained model
    if os.path.exists(model_save_path):
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            model.to(device)
            model.eval()
            print(f'Loaded model from {model_save_path}')
        except Exception as e:
            print(f'Error loading the model: {e}')
            return
    else:
        print(f'Model file {model_save_path} not found. Exiting evaluation.')
        return

    # Evaluation Metrics
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print('Evaluating model on test set...')
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc='Evaluation'):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    test_accuracy = 100 * correct / total
    print(f'Final Test Accuracy: {test_accuracy:.2f}%')

    # Classification Report
    print('\nClassification Report:')
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Save Classification Report
    report_df = pd.DataFrame(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)).transpose()
    report_df.to_csv(os.path.join(log_dir, 'classification_report_evaluate.csv'))
    print(f'Classification report saved to {os.path.join(log_dir, "classification_report_evaluate.csv")}')

    # Save Confusion Matrix Plot
    cm_plot_path = os.path.join(log_dir, 'confusion_matrix_evaluate.png')
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()
    print(f'Confusion matrix plot saved to {cm_plot_path}')

if __name__ == '__main__':
    main()


