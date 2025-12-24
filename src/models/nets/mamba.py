import json
from pathlib import Path
from typing import Literal

from mamba_ssm import Mamba2
from pydantic import BaseModel
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.models.schemas import MLPLayerConfig, build_mlp_from_config


class MambaConfig(BaseModel):
    # Always the same
    expand: int = 2
    d_conv: int = 4

    # Can vary
    d_state: Literal[64, 128] = 64
    headdim: Literal[64, 128] = 64


class MambaClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int,
            encoder_layers: list[MLPLayerConfig],
            mamba_config: dict | MambaConfig,
            embedding_dim: int,
            output_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        if isinstance(mamba_config, dict):
            mamba_config = MambaConfig(**mamba_config)

        # MLP Encoder Block
        self.mlp_encoder = build_mlp_from_config(encoder_layers, input_dim, embedding_dim)

        self.mamba = Mamba2(
            d_model=embedding_dim,
            **mamba_config.model_dump(),
        )

        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp_encoder(x)
        x = self.mamba(x)
        output = self.classifier(x[:, -1, :])
        return output


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']

        # Mamba params
        d_state = config['d_state']
        headdim = config['headdim']
        mamba_config = MambaConfig(
            d_state=d_state,
            headdim=headdim,
        )

        encoder_layers = [
            MLPLayerConfig(out_dim=config[f'encoder_dim_{idx}'], dropout=0.2)
            for idx in range(config['encoder_n_layers'])
        ]

        start_lr = config['lr']
    else:
        # Defaults
        embedding_dim = 32

        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=0.2),
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        mamba_config = MambaConfig()

        start_lr = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
            encoder_layers=encoder_layers,
            mamba_config=mamba_config,
            embedding_dim=embedding_dim,
            output_dim=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
        )
    )


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


def prep_mamba_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']

        # Mamba params
        d_state = config['d_state']
        headdim = config['headdim']
        mamba_config = MambaConfig(
            d_state=d_state,
            headdim=headdim,
        )

        encoder_layers = [
            MLPLayerConfig(out_dim=config[f'encoder_dim_{idx}'], dropout=0.2)
            for idx in range(config['encoder_n_layers'])
        ]

        start_lr = config['lr']
    else:
        embedding_dim = 32

        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=0.2),
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        mamba_config = MambaConfig()

        start_lr = 1e-2

    cfg = dict(
        model=dict(
            input_dim=input_dim,
            encoder_layers=encoder_layers,
            mamba_config=mamba_config,
            embedding_dim=embedding_dim,
            output_dim=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
        )
    )

    return cfg
