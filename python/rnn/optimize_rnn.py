from pathlib import Path

from torch.utils.data import DataLoader

from models.rnn import TabularRNN
from python.optimize_seq import optimize
from python.utils.optimization_utils import run_training_loop


def rnn_objective(trial, val_dataset, input_dim, num_steps, num_classes, batch_size, device, epochs, seed):
    '''Defines a single trial using a fixed train/validation split.'''

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=2, high=8)
    rnn_hidden_dim = 2 ** trial.suggest_int('rnn_hidden_dim_pow', low=4, high=6)
    mlp_n_layers = trial.suggest_int('mlp_n_layers', 1, 4)
    mlp_dims = [2 ** trial.suggest_int(f'mlp_dim_{i}_pow', low=4, high=8) for i in range(mlp_n_layers)]

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

    accuracy = run_training_loop(
        trial,
        model,
        val_loader,
        epochs,
        lr,
        device
    )

    return accuracy


def main():
    params_path = Path(f'data/params/rnn_optim')
    params_path.mkdir(parents=True, exist_ok=True)
    optimize(rnn_objective, params_path)


if __name__ == '__main__':
    main()
