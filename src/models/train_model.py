# src/models/train_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the output of the last time step
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(current_dir, '../../data/features/')
    features_file = os.path.join(features_dir, 'all_features.csv')

    # Load features
    df = pd.read_csv(features_file)
    print('Features loaded successfully.')

    # Preprocessing
    df = df.dropna()
    print('Missing values handled.')

    label_encoder = LabelEncoder()
    df['event_type_encoded'] = label_encoder.fit_transform(df['event_type'])
    num_classes = len(label_encoder.classes_)
    print('Labels encoded.')

    df = df.sort_values(by=['participant_id', 'timestamp'])
    print('Data sorted by participant and timestamp.')

    feature_columns = [
        'mean', 'std', 'var', 'skew', 'kurtosis', 'zero_crossing_rate',
        'bandpower_delta', 'relative_power_delta', 'bandpower_theta',
        'relative_power_theta', 'bandpower_alpha', 'relative_power_alpha',
        'bandpower_beta', 'relative_power_beta', 'bandpower_gamma',
        'relative_power_gamma'
    ]

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    print('Features scaled.')

    # Generate sequences
    sequence_length = 5
    sequences = []
    labels = []
    groups = []
    participants = df['participant_id'].unique()

    for participant in participants:
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

    # Split data
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(sequences, labels, groups=groups))

    X_train, X_test = sequences[train_idx], sequences[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print('Data split into training and testing sets.')

    # Create datasets and dataloaders
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    input_size = sequences.shape[2]
    hidden_size = 64
    model = LSTMModel(input_size, hidden_size, num_classes).to(device)
    print('Model initialized.')

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%')

    # Save the model
    model_save_path = os.path.join(current_dir, 'trained_lstm_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == '__main__':
    main()
