import numpy as np
import torch

from torch.utils.data import Dataset


class SequentialTabularDataset(Dataset):
    'Custom Dataset for sequences of tabular data.'
    def __init__(self, features, labels, device='cpu'):
        self.features = features
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return a sequence and its corresponding single label
        return torch.tensor(self.features[idx], dtype=torch.float32, device=self.device), \
               torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)



def create_sequences(df, group_cols, feature_cols, target_col, sequence_length):
    'Groups data and creates sliding window sequences.'
    sequences = []
    labels = []

    # Group by the columns that define an experiment
    grouped = df.groupby(group_cols)

    for _, group in grouped:
        # Skip experiments that are too short to form a sequence
        if len(group) < sequence_length:
            continue

        feature_data = group[feature_cols].values
        label_data = group[target_col].values

        # Use a sliding window to create sequences
        for i in range(len(group) - sequence_length + 1):
            seq = feature_data[i : i + sequence_length]
            # The label for the sequence is the label of its last element
            label = label_data[i + sequence_length - 1]

            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)