import argparse
import json
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset

from src.data.processing import create_sequences


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


def optimize(objective, params_path):
    args = parse_args()
    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device.upper()}\n', )

    data_path = Path('processed/processed')

    df_val = pd.read_csv(data_path / 'val.csv')

    with open(data_path / 'features.json', 'r') as f:
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
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    val_dataset = TensorDataset(X_val, y_val)

    input_dim = len(feature_cols)

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(
        partial(
            objective,
            val_dataset=val_dataset,
            input_dim=input_dim,
            num_steps=sequence_length,
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

    res = {
        'params':   best_params,
        'accuracy': best_acc,
    }
    with open(params_path / f'best_params_{args.filter}_{args.ds_type}.json', 'w') as f:
        json.dump(res, f, indent=2)
