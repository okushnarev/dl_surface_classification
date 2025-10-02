import argparse
import json
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.rnn import TabularRNN
from python.utils.dataset import SequentialTabularDataset, create_sequences


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--filter', type=str, default='no_filter', help='Filter used to create dataset')
    parser.add_argument('--ds_type', type=str, default='type_1', help='Dataset type')
    parser.add_argument('--use_cuda', action='store_true', help='Wheter to use CUDA')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for BPTT')
    parser.add_argument('--seed', type=int, default=69, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of jobs to perform search')
    return parser.parse_args()


def objective(trial, val_dataset, input_dim, num_classes, batch_size, device, epochs, seed):
    '''Defines a single trial using a fixed train/validation split.'''

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=2, high=8)
    rnn_hidden_dim = 2 ** trial.suggest_int('rnn_hidden_dim_pow', low=4, high=6)
    mlp_n_layers = trial.suggest_int('mlp_n_layers', 1, 4)
    mlp_dims = [2 **  trial.suggest_int(f'mlp_dim_{i}_pow', low=4, high=8) for i in range(mlp_n_layers)]

    # Create DataLoaders for this trial
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate Model and Optimizer
    model = TabularRNN(input_dim, mlp_dims, embedding_dim, rnn_hidden_dim, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss()

    # Training & Validation Loop
    # Train for a fixed number of epochs for each trial
    accuracy = 0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        epoch_loss = 0
        for sequences, labels in val_loader:
            # Data is already on the correct device from our custom Dataset
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step(epoch)

        accuracy = correct / total
        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy


def main():
    args = parse_args()
    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device.upper()}\n', )

    data_path = Path('data/datasets')

    df_val = pd.read_csv(data_path / 'val.csv')

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
    df_val[target_col] = label_encoder.fit_transform(df_val[target_col])

    num_classes = len(label_encoder.classes_)
    print(f'Classes found: {label_encoder.classes_}')

    # Scale features
    scaler = StandardScaler()
    df_val[feature_cols] = scaler.fit_transform(df_val[feature_cols])

    # Create the sequences
    X_val, y_val = create_sequences(df_val, group_cols, feature_cols, target_col, sequence_length)
    print(f'Created {len(X_val)} validation sequences.')

    # Create DataLoaders
    val_dataset = SequentialTabularDataset(X_val, y_val, device=device)

    input_dim = len(feature_cols)

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(
        partial(
            objective,
            val_dataset=val_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            batch_size=batch_size,
            device=device,
            epochs=args.epochs,
            seed=args.seed,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )


    trial_id = study.best_trial.number
    best_acc = study.best_trial.value
    _best_params = study.best_trial.params
    best_params = {}
    for param, val in _best_params.items():
        if 'pow' in param:
            name = param.replace('_pow', '')
            best_params[name] = 2 ** val
        else:
            best_params[param] = val


    print(f'Best trial: #{trial_id}')
    print(f'Best accuracy: {best_acc}')
    print(f'Best hyperparameters: {best_params}')

    params_path = Path(f'data/params/rnn_optim')
    params_path.mkdir(parents=True, exist_ok=True)

    res = {
        'params': best_params,
        'accuracy': best_acc,
    }
    with open(params_path / f'best_params_{args.filter}_{args.ds_type}.json', 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
