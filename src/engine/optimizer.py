import argparse
import inspect
import json
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from accelerate import find_executable_batch_size
from optuna import Trial
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.data.processing import chunk_split, create_sequences
from src.models.factory import get_model_components
from src.utils.paths import ProjectPaths
from src.utils.vars import CHUNK_COL


def add_optimizer_args(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group('Optimizer')
    group.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    group.add_argument('--output_file', type=str, default=None, help='Explicit output path for JSON params')
    group.add_argument('--exp_name', type=str, default=None, help='Experiment name')

    group.add_argument('--epochs', type=int, default=15)
    group.add_argument('--seq_len', type=int, default=10)
    group.add_argument('--val_size', type=float, default=0.1)
    group.add_argument('--batch_size', type=int, default=32768)
    group.add_argument('--num_workers', type=int, default=2)
    group.add_argument('--seed', type=int, default=69)
    group.add_argument('--filter', type=str, default='no_filter')
    group.add_argument('--ds_type', type=str, default='type_1')
    group.add_argument('--use_cuda', action='store_true')

    return parent_parser


def _execute_trial_loop(trial: Trial,
                        model,
                        train_dataset,
                        val_dataset,
                        epochs,
                        starting_batch_size,
                        lr,
                        weight_decay,
                        num_workers,
                        device):
    """
    Internal execution loop specifically for Optuna trials
    Handles forward passes, backprop, pruning, and reporting
    Handles CUDA OOM by reducing batch size
    """

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def train_loop(batch_size):
        # Create a DataLoader
        pin_memory = False if device == torch.device('cpu') else True
        _num_workers = 0 if device == torch.device('cpu') else num_workers
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                  num_workers=_num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=_num_workers)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        val_loss = np.inf

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for sequences, labels in train_loader:
                sequences = sequences.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(sequences)
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)
            scheduler.step(train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    return train_loop()


def generic_objective(trial, net_name, train_dataset, val_dataset, input_dim, num_classes, sequence_length, epochs,
                      starting_batch_size, num_workers, device):
    """
    The universal objective function used by Optuna
    Constructs the specific model using the Factory and runs the optimization loop
    """
    # Retrieve model components from the Factory
    components = get_model_components(net_name)
    ModelClass = components['class']
    get_params_func = components['optuna_params']

    # General params
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    weight_decay = 1e-4

    # Optuna model params
    model_kwargs = get_params_func(trial)
    # Check if the model accepts std_kwargs and update model_kwargs
    std_kwargs = dict(
        input_dim=input_dim,
        num_classes=num_classes,
        sequence_length=sequence_length,
    )
    model_signature = inspect.signature(ModelClass.__init__)
    for k, v in std_kwargs.items():
        if k in model_signature.parameters:
            model_kwargs[k] = v

    # Create model
    model = ModelClass(**model_kwargs).to(device)

    # Run the execution loop
    loss = _execute_trial_loop(
        trial=trial,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        starting_batch_size=starting_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        device=device
    )

    return loss


def run_optimization(args):
    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Starting Optimization for {args.nn_name} on {args.dataset}')
    print(f'Device: {device.type.upper()}')

    # Load the dataset configuration to understand metadata and feature groups
    with open(ProjectPaths.get_dataset_config_path(args.dataset), 'r') as f:
        dataset_config = json.load(f)

    # Extract configuration details
    group_cols = dataset_config['metadata']['group_cols'] + [CHUNK_COL]
    target_col = dataset_config['metadata']['target_col']
    feature_cols = dataset_config['features'][args.filter][args.ds_type]

    # Locate and load the proxy data
    data_dir = ProjectPaths.get_processed_data_dir(args.dataset)
    df_proxy = pd.read_csv(data_dir / 'proxy.csv')

    # Split proxy in train test
    df_train, df_val = chunk_split(
        df_proxy,
        group_cols=group_cols,
        target_col=target_col,
        # kwargs
        test_size=args.val_size,
        random_state=args.seed,
    )

    # Encode labels
    label_encoder = LabelEncoder()
    df_train[target_col] = label_encoder.fit_transform(df_train[target_col])
    df_val[target_col] = label_encoder.transform(df_val[target_col])
    num_classes = len(label_encoder.classes_)
    print(f'Classes: {label_encoder.classes_}')

    # Scale features using standard scaler
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])

    # Generate sequences for the model
    X_train, y_train = create_sequences(df_train, group_cols, feature_cols, target_col, args.seq_len)
    X_val, y_val = create_sequences(df_val, group_cols, feature_cols, target_col, args.seq_len)
    print(f'Created {len(X_train)} train and {len(X_val)} val sequences for optimization.')

    # Convert to PyTorch TensorDataset
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    # Initialize and run the Optuna study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3,
            max_resource=args.epochs,
        )
    )

    study.optimize(
        partial(
            generic_objective,
            net_name=args.nn_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_dim=len(feature_cols),
            num_classes=num_classes,
            sequence_length=args.seq_len,
            epochs=args.epochs,
            starting_batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        ),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    # Process results from the best trial
    trial = study.best_trial
    print(f'Best trial: #{trial.number} | Loss: {trial.value}')

    # Convert power-of-2 parameters (suffix _pow) to their actual values
    best_params_processed = {}
    for param, val in trial.params.items():
        if 'pow' in param:
            name = param.replace('_pow', '')
            best_params_processed[name] = 2 ** val
        else:
            best_params_processed[param] = val

    print(f'Best Hyperparams: {best_params_processed}')

    # Prepare result dictionary
    res = {
        'params': best_params_processed,
        'loss':   trial.value,
    }

    # Save path
    if args.output_file:
        save_path = Path(args.output_file)
    else:
        # Auto-generate path based on conventions if no explicit output provided
        name = args.exp_name if args.exp_name else f'{args.filter}_{args.ds_type}'
        save_path = ProjectPaths.get_params_path(args.nn_name, args.config_name, name)

    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Saving best params to: {save_path}')
    with open(save_path, 'w') as f:
        json.dump(res, f, indent=2)
