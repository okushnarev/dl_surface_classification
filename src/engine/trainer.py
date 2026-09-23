import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import find_executable_batch_size
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.data.processing import create_sequences
from src.models.factory import get_model_components
from src.utils.io import load_checkpoint, save_checkpoint
from src.utils.paths import ProjectPaths
from src.utils.seed import seed_everything, seed_worker
from src.utils.vars import CHUNK_COL


def add_trainer_args(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group('Trainer')

    group.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    group.add_argument('--filter', type=str, default='no_filter', help='Filter used to create dataset')
    group.add_argument('--batch_size', type=int, default=32768)
    group.add_argument('--feature_set', type=str, default='type_1', help='Feature set')
    group.add_argument('--use_cuda', action='store_true', help='Wheter to use CUDA')
    group.add_argument('--seq_len', type=int, default=10, help='Sequence length for BPTT')
    group.add_argument('--param_file', type=str, default=None, help='Path to JSON config')
    group.add_argument('--val_every', type=int, default=5, help='Val model every N epochs')
    group.add_argument('--save_every', type=int, default=10, help='Save model every N epochs')
    group.add_argument('--early_stop_patience', type=int, default=7,
                       help='Early stop after no loss improvement after N val epochs')
    group.add_argument('--exp_name', type=str, default=None, help='Experiment name for a run')
    group.add_argument('--num_workers', type=int, default=1, help='Number of workers for Dataloader')
    group.add_argument('--restart_behavior', choices=['resume', 'restart'], default='restart',
                       help='Resume loads checkpoint and continue training\n Restart overwrites everything')
    group.add_argument('--seed', type=int, default=69, help='Seed everything integer')
    return parent_parser


def train_model(args):
    # Setup
    seed_everything(args.seed)
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device.type.upper()}')

    # Paths
    data_dir = ProjectPaths.get_processed_data_dir(args.dataset)

    # Define run name
    run_name = args.exp_name or datetime.now().strftime('%Y%m%d_%H%M%S')

    ckpt_path = ProjectPaths.get_run_dir(args.config_name, run_name)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Load Data
    print(f'Loading data from {data_dir}')
    df_train = pd.read_csv(data_dir / 'train.csv')
    df_val = pd.read_csv(data_dir / 'val.csv')

    with open(ProjectPaths.get_dataset_config_path(args.dataset), 'r') as f:
        datasets_cfg = json.load(f)

    # Params
    group_cols = datasets_cfg['metadata']['group_cols'] + [CHUNK_COL]
    target_col = datasets_cfg['metadata']['target_col']
    feature_cols = datasets_cfg['features'][args.filter][args.feature_set]

    # Encode and scale
    label_encoder = LabelEncoder()
    df_train[target_col] = label_encoder.fit_transform(df_train[target_col])
    df_val[target_col] = label_encoder.transform(df_val[target_col])

    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])

    # Save scaler and label encoder
    joblib.dump(scaler, ckpt_path / 'scaler.joblib')
    joblib.dump(label_encoder, ckpt_path / 'label_encoder.joblib')

    # Create sequences
    X_train, y_train = create_sequences(df_train, group_cols, feature_cols, target_col, args.seq_len)
    X_val, y_val = create_sequences(df_val, group_cols, feature_cols, target_col, args.seq_len)
    print(f'Created {len(X_train)} training sequences and {len(X_val)} val sequences.')

    # Create datasets
    X_train = torch.tensor(X_train, dtype=torch.bfloat16)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)

    X_val = torch.tensor(X_val, dtype=torch.bfloat16)
    y_val = torch.tensor(y_val, dtype=torch.long)
    val_dataset = TensorDataset(X_val, y_val)

    # Initialize model
    components = get_model_components(args.nn_name)
    prep_cfg_func = components['prep_config']
    ModelClass = components['class']

    # Parse config file to get model args
    if args.param_file:
        # Explicit path provided via CLI/YAML
        cfg_path = Path(args.param_file)
    elif args.exp_name:
        # Auto-resolve based on experiment name
        cfg_path = ProjectPaths.get_params_path(args.nn_name, args.config_name, args.exp_name)
    else:
        # Fallback/Error. Shouldn't happen in proper runs
        cfg_path = None
        print('Warning: No param_file and no exp_name. Using model defaults.')

    # Check for a valid config path
    if cfg_path and not cfg_path.exists():
        raise FileNotFoundError(f'Parameter file not found: {cfg_path}')

    cfg = prep_cfg_func(
        cfg_path,
        input_dim=len(feature_cols),
        num_classes=len(label_encoder.classes_),
        sequence_length=args.seq_len
    )

    model = ModelClass(**cfg['model']).to(device, dtype=torch.bfloat16)

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def train_loop(batch_size):
        nonlocal model
        # Create dataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False if device == torch.device('cpu') else True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=1,
            worker_init_fn=seed_worker,
            shuffle=False

        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['optimizer']['start_lr'],
            weight_decay=cfg['optimizer']['weight_decay']
        )
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, min_lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        start_epoch = 0
        best_val_loss = float('inf')
        early_stop_counter = 0
        val_acc = 0

        if args.restart_behavior == 'resume' and (ckpt_path / 'last.pt').exists():
            print('Resuming training from checkpoint')
            checkpoint = load_checkpoint(model, optimizer, ckpt_path / 'last.pt', device)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('best_acc', 0)

        print(f'\n--- Starting Training: {args.nn_name.upper()}. Batch size: {batch_size} ---')

        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0
            for sequences, labels in train_loader:
                sequences = sequences.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(sequences)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss /= len(train_loader)

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{args.epochs}] Loss: {epoch_loss:.6f} LR: {current_lr:.2e}')

            # Val
            if epoch % args.val_every == 0 or epoch == args.epochs - 1:
                model.eval()
                val_loss = 0
                correct, total = 0, 0
                with torch.no_grad():
                    for sequences, labels in val_loader:
                        sequences, labels = sequences.to(device), labels.to(device)
                        outputs = model(sequences)
                        predicted = torch.argmax(outputs, 1)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_acc = correct / total
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                print(f'  Val loss: {val_loss:.6f}')
                print(f'  Accuracy on the val set: {100 * val_acc:.2f} %')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    print('  The best vall loss found')
                    save_checkpoint(model, optimizer, epoch, epoch_loss, val_acc, ckpt_path / 'best.pt')
                else:
                    early_stop_counter += 1

                if early_stop_counter >= args.early_stop_patience:
                    print('  Early stopping triggered')
                    save_checkpoint(model, optimizer, epoch, epoch_loss, val_acc, ckpt_path / 'last.pt')
                    break

            if epoch % args.save_every == 0 or epoch == args.epochs - 1:
                save_checkpoint(model, optimizer, epoch, epoch_loss, val_acc, ckpt_path / 'last.pt')

    train_loop()
