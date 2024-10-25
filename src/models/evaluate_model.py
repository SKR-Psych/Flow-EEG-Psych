# src/models/evaluate_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Custom Dataset Class
class EEGDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = sequences
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return X_tensor, y_tensor

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Take the output from the last time step
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, num_classes)
        return out

def main():
    # Set random seed
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
    # Handle missing values
    initial_shape = df.shape
    df = df.dropna()
    final_shape = df.shape
    print(f'Missing values handled. Dropped {initial_shape[0] - final_shape[0]} rows.')

    # Encode labels
    label_encoder = LabelEncoder()
    df['event_type_encoded'] = label_encoder.fit_transform(df['event_type'])
    num_classes = len(label_encoder.classes_)
    print(f'Labels encoded. Classes: {label_encoder.classes_}')

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

    # Generate sequences
    sequence_length = 5  # Number of time steps per sequence
    sequences = []
    labels = []
    groups = []  # To keep track of participant IDs for GroupShuffleSplit
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
                lbl = label[i+sequence_length-1]  # Label for the last time step in the sequence
                sequences.append(seq)
                labels.append(lbl)
                groups.append(participant)

    sequences = np.array(sequences)
    labels = np.array(labels)
    print(f'Total sequences generated: {sequences.shape[0]}')

    # Split data while avoiding data leakage (GroupShuffleSplit by participant)
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(sequences, labels, groups=groups))

    X_train, X_test = sequences[train_idx], sequences[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(f'Data split into training and testing sets.')
    print(f'Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}')

    # Create Datasets and DataLoaders
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)

    # Initialize the model
    input_size = sequences.shape[2]  # Number of features
    hidden_size = 64
    num_layers = 2
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout=0.5).to(device)
    print('Model initialized.')

    # Load the trained model
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
        model.eval()
        print(f'Loaded model from {model_save_path}')
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
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Save Classification Report
    report_df = pd.DataFrame(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True)).transpose()
    report_df.to_csv(os.path.join(log_dir, 'classification_report_evaluate.csv'))
    print(f'Classification report saved to {os.path.join(log_dir, "classification_report_evaluate.csv")}')
    
    # Save Confusion Matrix Plot
    cm_plot_path = os.path.join(log_dir, 'confusion_matrix_evaluate.png')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f'Confusion matrix plot saved to {cm_plot_path}')

if __name__ == '__main__':
    main()
