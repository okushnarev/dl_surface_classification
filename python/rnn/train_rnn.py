import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.rnn import TabularRNN
from python.utils.dataset import SequentialTabularDataset, create_sequences
from python.utils.save_load import load_checkpoint, save_checkpoint
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--filter', type=str, default='no_filter', help='Filter used to create dataset')
    parser.add_argument('--ds_type', type=str, default='type_1', help='Dataset type')
    parser.add_argument('--use_cuda', action='store_true', help='Wheter to use CUDA')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for BPTT')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--test_every', type=int, default=20, help='Test model every N epochs')
    parser.add_argument('--save_every', type=int, default=50, help='Save model every N epochs')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for a run')
    parser.add_argument('--restart_behavior', choices=['resume', 'restart'], default='restart',
                        help='Resume loads checkpoint and continue training\n Restart overwrites everything')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device.upper()}\n', )

    data_path = Path('data/datasets')

    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = Path('data/runs') / exp_name
    ckpt_path.mkdir(parents=True, exist_ok=True)

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

    # Save scaler
    with open(ckpt_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

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
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']
        rnn_hidden_dim = config['rnn_hidden_dim']
        mlp_hidden_dims = []
        for idx in range(config['mlp_n_layers']):
            mlp_hidden_dims.append(config[f'mlp_dim_{idx}'])

        start_lr = config['lr']
    else:
        embedding_dim = 32
        mlp_hidden_dims = [64, 16]
        rnn_hidden_dim = 64
        start_lr = 1e-2

    model = TabularRNN(
        input_dim=input_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        embedding_dim=embedding_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        num_classes=num_classes,
        device=device
    )

    # Training
    num_epochs = args.epochs

    # Load model and optimizer from checkpoint if allowed by behavior
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    start_epoch = 0
    if args.restart_behavior == 'resume' and (ckpt := ckpt_path / 'checkpoint.pth').exists():
        print('Resuming training from checkpoint')
        start_epoch = load_checkpoint(model, optimizer, ckpt, device)

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    print('\n--- Starting Training ---')
    best_acc = 0
    for epoch in range(start_epoch, num_epochs):
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

        # Test every N epochs
        if epoch % args.test_every == 0:
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for sequences, labels in test_loader:
                    outputs = model(sequences)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = correct / total
            print(f'Accuracy on the test set: {100 * test_acc:.2f} %')
            # Save model if the best accuracy achieved
            if test_acc > best_acc:
                best_acc = test_acc
                print('The best accuracy found')
                save_checkpoint(model, optimizer, epoch, loss, ckpt_path / 'best.pt')

        # Save checkpoint every N epoch
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss, ckpt_path / 'last.pt')

    # Test
    print('\n--- Starting Evaluation ---')
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for sequences, labels in test_loader:
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on the test set: {100 * correct / total:.2f} %')
    save_checkpoint(model, optimizer, epoch, loss, ckpt_path / 'last.pt')


if __name__ == '__main__':
    main()
