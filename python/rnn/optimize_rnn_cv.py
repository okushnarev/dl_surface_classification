import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset

from models.rnn import TabularRNN
from python.utils.dataset import create_sequences


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
    parser.add_argument('--cv_splits', type=int, default=3, help='Number of CV splits')
    return parser.parse_args()


def objective(trial, full_train_dataset, input_dim, num_classes, batch_size, cv_splits, device, epochs, seed):
    '''Defines a single trial for hyperparameter optimization with 3-fold CV.'''

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=2, high=8)
    rnn_hidden_dim = 2 ** trial.suggest_int('rnn_hidden_dim_pow', low=4, high=6)
    mlp_n_layers = trial.suggest_int('mlp_n_layers', low=1, high=4)
    mlp_dims = [2 ** trial.suggest_int(f'mlp_dim_{i}_pow', low=4, high=8) for i in range(mlp_n_layers)]

    # Cross-Validation Setup
    # StratifiedKFold is good for classification to preserve class distribution.
    kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    fold_accuracies = []

    all_labels = full_train_dataset.labels.cpu().numpy() if device == 'gpu' else full_train_dataset.labels

    global_step = 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(full_train_dataset)), all_labels)):
        # Create datasets and dataloaders for the current fold
        train_subset = Subset(full_train_dataset, train_idx)
        val_subset = Subset(full_train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Instantiate Model and Optimizer for this Fold
        model = TabularRNN(input_dim, mlp_dims, embedding_dim, rnn_hidden_dim, num_classes).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
        criterion = nn.CrossEntropyLoss()

        # Training & Validation Loop for the Fold
        # We train for a smaller number of epochs during HPO search to be faster.
        for epoch in range(epochs):
            model.train()
            for sequences, labels in train_loader:
                outputs = model(sequences)
                loss = criterion(outputs, labels.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation and Pruning
            model.eval()
            correct, total = 0, 0
            val_loss = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    outputs = model(sequences)
                    val_loss += criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.squeeze()).sum().item()

            scheduler.step(val_loss)

            accuracy = correct / total

            # Report intermediate accuracy to Optuna
            trial.report(accuracy, global_step)
            global_step += 1

            # Handle pruning: Check if the trial should be stopped early
            if trial.should_prune():
                raise optuna.TrialPruned()

        fold_accuracies.append(accuracy)

    # Return the mean accuracy across all folds
    return np.mean(fold_accuracies)


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
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    val_dataset = TensorDataset(X_val, y_val)

    input_dim = len(feature_cols)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(
        partial(
            objective,
            full_train_dataset=val_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            batch_size=batch_size,
            cv_splits=args.cv_splits,
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
        'params':   best_params,
        'accuracy': best_acc,
    }
    with open(params_path / f'best_params_cv.json', 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
