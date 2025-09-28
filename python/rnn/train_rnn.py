import argparse

import numpy as np
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.rnn import TabularRNN
from python.utils.dataset import SequentialTabularDataset, create_sequences

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--filter', type=str, default='no_filter', help='Filter used to create dataset')
    parser.add_argument('--ds_type', type=str, default='type_1', help='Dataset type')
    parser.add_argument('--use_cuda', action='store_true', help='Wheter to use CUDA')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for BPTT')
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device.upper()}\n',)

    data_path = Path('data/datasets')

    df_train = pd.read_csv(data_path / 'train.csv')
    df_test = pd.read_csv(data_path / 'test.csv')

    with open(data_path / 'datasets_features.json', 'r') as f:
        datasets = json.load(f)

    # Params
    group_cols = ['surf', 'movedir', 'speedamp']
    feature_cols = datasets[args.filter][args.ds_type]
    target_col = 'surf'
    sequence_length = args.seq_len

    batch_size = 2 ** 15

    # Encode labels
    label_encoder = LabelEncoder()
    df_train[target_col] = label_encoder.fit_transform(df_train[target_col])
    df_test[target_col] = label_encoder.transform(df_test[target_col])

    num_classes = len(label_encoder.classes_)
    print(f'Classes found: {label_encoder.classes_}')

    # Scale features
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    # Create the sequences
    X_train, y_train = create_sequences(df_train, group_cols, feature_cols, target_col, sequence_length)
    X_test, y_test = create_sequences(df_test, group_cols, feature_cols, target_col, sequence_length)
    print(f'Created {len(X_train)} training sequences and {len(X_test)} test sequences.')

    # Create DataLoaders
    train_dataset = SequentialTabularDataset(X_train, y_train, device=device)
    test_dataset = SequentialTabularDataset(X_test, y_test, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    input_dim = len(feature_cols)
    embedding_size = 32
    mlp_hidden_dims = [64, 16]
    rnn_hidden_dim = 64

    model = TabularRNN(
        input_dim=input_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        embedding_dim=embedding_size,
        rnn_hidden_dim=rnn_hidden_dim,
        num_classes=num_classes,
        device=device
    )

    # Training
    num_epochs = 100
    start_lr = 1e-2

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    print('\n--- Starting Training ---')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for sequences, labels in train_loader:
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(epoch_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.10f}, LR: {current_lr:.3e}')

    # Eval
    print('\n--- Starting Evaluation ---')
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        score, cnt = [], []
        for sequences, labels in test_loader:
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            score.append(accuracy_score(labels.cpu(), predicted.cpu()))
            cnt.append(labels.size(0))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on the test set: {100 * correct / total:.2f} %')
    print(f'Score (sklearn): {100 * np.average(score, weights=cnt):.2f} %')


if __name__ == '__main__':
    main()
