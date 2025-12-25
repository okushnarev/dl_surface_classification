import argparse
import json
from functools import partial
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.data.processing import create_sequences
from src.models.factory import get_model_components
from src.utils.paths import ProjectPaths


def add_optimizer_args(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group('Optimizer')
    group.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    group.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    group.add_argument('--output_file', type=str, default=None, help='Explicit output path for JSON params')
    group.add_argument('--exp_name', type=str, default=None, help='Experiment name')

    # Common training args that affect optimization
    group.add_argument('--epochs', type=int, default=15)
    group.add_argument('--seq_len', type=int, default=10)
    group.add_argument('--batch_size', type=int, default=32768)
    group.add_argument('--seed', type=int, default=69)
    group.add_argument('--filter', type=str, default='no_filter')
    group.add_argument('--ds_type', type=str, default='type_1')
    group.add_argument('--use_cuda', action='store_true')

    return parent_parser


def _execute_trial_loop(trial, model, data_loader, epochs, lr, device):
    """
    Internal execution loop specifically for Optuna trials
    Handles forward passes, backprop, pruning, and reporting
    Does not save checkpoints or logs to disk to maximize speed
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        epoch_loss = 0

        for sequences, labels in data_loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(sequences)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(data_loader)
        scheduler.step(epoch_loss)

        accuracy = correct / total

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy


def generic_objective(trial, net_name, val_dataset, input_dim, num_classes, seq_len, batch_size, device, epochs):
    """
    The universal objective function used by Optuna
    Constructs the specific model using the Factory and runs the optimization loop
    """
    # Retrieve model components from the Factory
    components = get_model_components(net_name)
    ModelClass = components['class']
    get_params_func = components['optuna_params']

    # Generate hyperparameters from the search space
    config = get_params_func(trial)

    # Read and Update the model params
    model_kwargs = config['model_kwargs']
    model_kwargs['input_dim'] = input_dim
    model_kwargs['num_classes'] = num_classes

    model_kwargs['sequence_length'] = seq_len

    # Create model
    model = ModelClass(**model_kwargs).to(device)

    # Create a DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Run the execution loop
    accuracy = _execute_trial_loop(
        trial=trial,
        model=model,
        data_loader=val_loader,
        epochs=epochs,
        lr=config['lr'],
        device=device
    )

    return accuracy


def run_optimization(args):
    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    print(f'Starting Optimization for {args.nn_name} on {args.dataset}')
    print(f'Device: {device.upper()}')

    # Locate and load the validation data
    data_dir = ProjectPaths.get_processed_data_dir(args.dataset)
    df_val = pd.read_csv(data_dir / 'val.csv')

    # Load the dataset configuration to understand metadata and feature groups
    with open(ProjectPaths.get_dataset_config_path(args.dataset), 'r') as f:
        dataset_config = json.load(f)

    # Extract configuration details
    group_cols = dataset_config['metadata']['group_cols']
    target_col = dataset_config['metadata']['target_col']
    feature_cols = dataset_config['features'][args.filter][args.ds_type]

    # Encode labels
    label_encoder = LabelEncoder()
    df_val[target_col] = label_encoder.fit_transform(df_val[target_col])
    num_classes = len(label_encoder.classes_)
    print(f'Classes: {label_encoder.classes_}')

    # Scale features using standard scaler
    scaler = StandardScaler()
    df_val[feature_cols] = scaler.fit_transform(df_val[feature_cols])

    # Generate sequences for the model
    X_val, y_val = create_sequences(df_val, group_cols, feature_cols, target_col, args.seq_len)
    print(f'Created {len(X_val)} sequences for optimization.')

    # Convert to PyTorch TensorDataset
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    # Initialize and run the Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

    study.optimize(
        partial(
            generic_objective,
            net_name=args.nn_name,
            val_dataset=val_dataset,
            input_dim=len(feature_cols),
            num_classes=num_classes,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
            epochs=args.epochs
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    # Process results from the best trial
    trial = study.best_trial
    print(f'Best trial: #{trial.number} | Accuracy: {trial.value}')

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
        'params':   best_params_processed,
        'accuracy': trial.value,
    }

    # Save path
    if args.output_file:
        save_path = Path(args.output_file)
    else:
        # Auto-generate path based on conventions if no explicit output provided
        name = args.exp_name if args.exp_name else f'{args.filter}_{args.ds_type}'
        save_path = ProjectPaths.get_params_path(args.nn_name, args.dataset, name)

    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Saving best params to: {save_path}')
    with open(save_path, 'w') as f:
        json.dump(res, f, indent=2)