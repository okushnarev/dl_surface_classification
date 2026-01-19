import json
from pathlib import Path
from typing import Literal

from mamba_ssm import Mamba2
from pydantic import BaseModel
from torch import Tensor, nn

from eval_rnn import embedding_dim
from src.models.schemas import MLPLayerConfig, build_funnel_dims, build_mlp_from_config


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
            num_classes: int):
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

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp_encoder(x)
        x = self.mamba(x)
        output = self.classifier(x[:, -1, :])
        return output


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        # Mamba params
        d_state = config['d_state']
        headdim = config['headdim']
        mamba_config = MambaConfig(
            d_state=d_state,
            headdim=headdim,
        )

        dropout = config.get('dropout', 0.2)
        encoder_n_layers = config['encoder_n_layers']
        if 'encoder_initial_dim' in config:
            # New funnel approach
            encoder_initial_dim = config['encoder_initial_dim']
            encoder_expand_factor = config['encoder_expand_factor']

            encoder_dims = build_funnel_dims(encoder_initial_dim, encoder_n_layers, encoder_expand_factor)
            embedding_dim = encoder_dims[-1]
        else:
            # Backward compatibility
            encoder_dims = [config[f'mlp_dim_{idx}'] for idx in range(encoder_n_layers)]
            embedding_dim = config['embedding_dim']

        encoder_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in encoder_dims]

        start_lr = config['lr']
        weight_decay = config.get('weight_decay', 1e-4)
    else:
        # Defaults
        embedding_dim = 32

        dropout = 0.2
        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=dropout),
            MLPLayerConfig(out_dim=32, dropout=dropout),
        ]

        mamba_config = MambaConfig()

        start_lr = 1e-2
        weight_decay = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
            encoder_layers=encoder_layers,
            mamba_config=mamba_config,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
            weight_decay=weight_decay,
        )
    )


def get_optuna_params(trial):
    dropout = 0.2

    # Mamba config
    d_state = 2 ** trial.suggest_int('d_state_pow', low=6, high=7)
    headdim = 2 ** trial.suggest_int('headdim_pow', low=6, high=7)

    mamba_config = MambaConfig(
        d_state=d_state,
        headdim=headdim,
    )

    # Encoder config
    encoder_n_layers = trial.suggest_int('encoder_n_layers', 1, 4)
    encoder_initial_dim = 2 ** trial.suggest_int('encoder_initial_dim_pow', low=2, high=5)
    encoder_expand_factor = 2 ** trial.suggest_int('encoder_expand_factor_pow', low=0, high=2)
    encoder_dims = build_funnel_dims(encoder_initial_dim, encoder_n_layers, encoder_expand_factor)
    encoder_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in encoder_dims]

    # Embedding dim
    embedding_dim = encoder_dims[-1]

    return dict(
        encoder_layers=encoder_layers,
        mamba_config=mamba_config,
        embedding_dim=embedding_dim,
    )
