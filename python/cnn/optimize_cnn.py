from pathlib import Path

import optuna
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.cnn import CNNLayerConfig, MLPLayerConfig
from python.cnn.train_cnn import CNNTrainWrapper
from python.optimize_seq import optimize


def cnn_objective(trial, val_dataset, input_dim, num_steps, num_classes, batch_size, device, epochs, seed):
    """Defines a single trial using a fixed train/validation split."""

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    cnn_n_layers = trial.suggest_int('cnn_n_layers', 1, 3)
    cnn_channels = [2 ** trial.suggest_int(f'cnn_out_ch_{i}_pow', low=4, high=8) for i in range(cnn_n_layers)]
    mlp_n_layers = trial.suggest_int('mlp_n_layers', 1, 4)
    mlp_dims = [2 ** trial.suggest_int(f'mlp_dim_{i}_pow', low=4, high=8) for i in range(mlp_n_layers)]

    # Create DataLoaders for this trial
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate Model and Optimizer
    cnn_configs = [
        CNNLayerConfig(out_channels=ch, kernel_size=3)
        for ch in cnn_channels
    ]
    mlp_configs = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in mlp_dims
    ]
    model = CNNTrainWrapper(
        input_dim=input_dim,
        num_steps=num_steps,
        cnn_configs=cnn_configs,
        mlp_configs=mlp_configs,
        num_classes=num_classes
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss()

    accuracy = 0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        epoch_loss = 0
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

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
    params_path = Path(f'data/params/cnn_optim')
    params_path.mkdir(parents=True, exist_ok=True)
    optimize(cnn_objective, params_path)


if __name__ == '__main__':
    main()
