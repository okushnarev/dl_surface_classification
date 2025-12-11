from pathlib import Path

from torch.utils.data import DataLoader

from models.cnn import MLPLayerConfig
from models.configs.mamba_config import MambaConfig
from models.mamba import MambaClassifier
from python.optimize_seq import optimize
from python.utils.optimization_utils import run_training_loop


def mamba_objective(trial, val_dataset, input_dim, num_steps, num_classes, batch_size, device, epochs, seed):
    """Defines a single trial using a fixed train/validation split."""

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=6, high=9)

    d_state = 2 ** trial.suggest_int('d_state_pow', low=6, high=7)
    headdim = 2 ** trial.suggest_int('headdim_pow', low=6, high=7)
    mamba_config = MambaConfig(
        d_state=d_state,
        headdim=headdim,
    )

    encoder_n_layers = trial.suggest_int('encoder_n_layers', 1, 4)
    encoder_dims = [2 ** trial.suggest_int(f'encoder_dim_{i}_pow', low=4, high=8) for i in range(encoder_n_layers)]

    encoder_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in encoder_dims
    ]

    # Create DataLoaders for this trial
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    # Instantiate Model and Optimizer
    model = MambaClassifier(
        input_dim=input_dim,
        encoder_layers=encoder_layers,
        mamba_config=mamba_config,
        embedding_dim=embedding_dim,
        output_dim=num_classes,
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
    params_path = Path(f'data/params/transformer_optim')
    params_path.mkdir(parents=True, exist_ok=True)
    optimize(mamba_objective, params_path)


if __name__ == '__main__':
    main()
