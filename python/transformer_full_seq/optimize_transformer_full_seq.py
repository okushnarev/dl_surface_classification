from pathlib import Path

from torch.utils.data import DataLoader

from models.cnn import MLPLayerConfig
from models.transformer_full_seq import TransformerFullSeq
from python.optimize_seq import optimize
from python.utils.optimization_utils import run_training_loop


def transformer_full_seq_objective(trial, val_dataset, input_dim, num_steps, num_classes, batch_size, device, epochs, seed):
    """Defines a single trial using a fixed train/validation split."""

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2)
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=2, high=7)
    num_transformer_heads = 2 ** trial.suggest_int('num_transformer_heads_pow', low=0, high=2)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', low=1, high=4)

    classification_n_layers = trial.suggest_int('classification_n_layers', 1, 4)
    classification_dims = [2 ** trial.suggest_int(f'classification_dim_{i}_pow', low=4, high=8) for i in
                           range(classification_n_layers)]
    classification_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in classification_dims
    ]

    # Create DataLoaders for this trial
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    # Instantiate Model and Optimizer
    model = TransformerFullSeq(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,
        classification_layers=classification_layers,
        num_classes=num_classes,
    ).to(device)

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
    params_path = Path(f'data/params/transformer_full_seq_optim')
    params_path.mkdir(parents=True, exist_ok=True)
    optimize(transformer_full_seq_objective, params_path)


if __name__ == '__main__':
    main()
