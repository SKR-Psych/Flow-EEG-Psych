# src/models/dataset.py

import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        Args:
            sequences (numpy.ndarray): Array of shape (num_samples, sequence_length, num_features)
            labels (numpy.ndarray): Array of shape (num_samples,)
        """
        self.X = sequences
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return X_tensor, y_tensor

