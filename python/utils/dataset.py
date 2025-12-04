import numpy as np
import torch

from torch.utils.data import Dataset


class SequentialTabularDataset(Dataset):
    'Custom Dataset for sequences of tabular data.'

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return a sequence and its corresponding single label
        return torch.tensor(self.features[idx], dtype=torch.float32), \
            torch.tensor(self.labels[idx], dtype=torch.long)


def create_sequences(df, group_cols, feature_cols, target_col, sequence_length, info_cols=None):
    'Groups data and creates sliding window sequences.'
    sequences = []
    labels = []
    infos = []

    # Group by the columns that define an experiment
    grouped = df.groupby(group_cols)

    for _, group in grouped:
        # Skip experiments that are too short to form a sequence
        if len(group) < sequence_length:
            continue

        feature_data = group[feature_cols].values
        label_data = group[target_col].values
        info_data = group[info_cols].values if info_cols else None

        # Use a sliding window to create sequences
        for i in range(len(group) - sequence_length + 1):
            seq = feature_data[i: i + sequence_length]
            # The label for the sequence is the label of its last element
            label = label_data[i + sequence_length - 1]

            sequences.append(seq)
            labels.append(label)

            if info_cols:
                info = info_data[i + sequence_length - 1]
                infos.append(info)

    if info_cols:
        return np.array(sequences), np.array(labels), np.array(infos)
    else:
        return np.array(sequences), np.array(labels)
