import torch
import optuna
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import Module


def run_training_loop(
        trial: optuna.Trial,
        model: Module,
        data_loader: DataLoader,
        epochs: int,
        lr: float,
        device: str,
):
    """
    Generic training loop that handles forward/backward passes, Optuna reporting, and pruning
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    accuracy = 0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        epoch_loss = 0

        for sequences, labels in data_loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(data_loader)
        scheduler.step(epoch_loss)

        accuracy = correct / total

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy
