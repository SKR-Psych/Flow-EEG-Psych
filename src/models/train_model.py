# src/models/train_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import LSTMModel
from dataset import EEGDataset
from utils import set_seed

# Implementing Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        else:
            BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                F_loss = self.alpha * F_loss
            else:
                alpha = self.alpha[targets].view(-1, 1)
                F_loss = alpha * F_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

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
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    batch_size = 64  # Increased batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Compute class weights to handle class imbalance
    # Only include classes present in y_train
    classes_in_train = np.unique(y_train)
    class_weights_list = compute_class_weight(class_weight='balanced',
                                             classes=classes_in_train,
                                             y=y_train)
    # Initialize all class weights to 1.0
    class_weights = np.ones(num_classes)
    # Assign computed weights to the corresponding classes
    class_weights[classes_in_train] = class_weights_list
    # Convert to torch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f'Class weights: {class_weights}')

    # Initialize the model
    input_size = sequences.shape[2]
    hidden_size = 128  # Increased hidden size
    num_layers = 3      # Increased number of layers
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout=0.5).to(device)
    print('Model initialized.')

    # Define loss and optimizer
    # Using Focal Loss to address class imbalance
    criterion = FocalLoss(alpha=class_weights, gamma=2, logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # Training parameters
    num_epochs = 50
    best_val_accuracy = 0.0
    patience = 5  # For early stopping
    counter = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

            progress_bar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / total_train
        train_accuracy = 100 * correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader, desc='Validation', leave=False):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += y_batch.size(0)
                correct_val += (predicted == y_batch).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / total_val
        val_accuracy = 100 * correct_val / total_val

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check for improvement
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            counter = 0
            improved = True
        else:
            counter += 1
            improved = False

        # Logging
        print(f'Epoch [{epoch}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        if improved:
            print('Validation accuracy improved. Model saved.')
        else:
            print('No improvement in validation accuracy.')
            if counter >= patience:
                print('Early stopping triggered.')
                break

    # Load the best model for final evaluation
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()
        print(f'Loaded best model from {model_save_path}')
    except Exception as e:
        print(f'Error loading the best model: {e}')
        return

    # Final Evaluation on Test Set
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc='Final Evaluation'):
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


