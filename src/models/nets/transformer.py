import json
from pathlib import Path

import torch
import torch.nn as nn

from src.models.schemas import MLPLayerConfig, build_mlp_from_config


class Transformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_layers: list[MLPLayerConfig],
                 embedding_dim: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 classification_layers: list[MLPLayerConfig],
                 num_classes: int):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = build_mlp_from_config(encoder_layers, input_dim, embedding_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_transformer_heads,
            batch_first=True,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_transformer_layers
        )

        # Classification head
        self.classifier = build_mlp_from_config(classification_layers, embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.encoder(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = self.classifier(x[:, 0, :])
        return x


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']
        num_transformer_heads = config['num_transformer_heads']
        num_transformer_layers = config['num_transformer_layers']

        encoder_layers = [
            MLPLayerConfig(out_dim=config[f'encoder_dim_{idx}'], dropout=0.2)
            for idx in range(config['encoder_n_layers'])
        ]

        classification_layers = [
            MLPLayerConfig(out_dim=config[f'classification_dim_{idx}'], dropout=0.2)
            for idx in range(config['classification_n_layers'])
        ]

        start_lr = config['lr']
    else:
        # Defaluts
        embedding_dim = 32
        num_transformer_heads = 1
        num_transformer_layers = 1

        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=0.2),
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        classification_layers = [
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        start_lr = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
            encoder_layers=encoder_layers,
            embedding_dim=embedding_dim,
            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers,
            classification_layers=classification_layers,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
        )
    )


def get_optuna_params(trial):
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=4, high=8)
    num_transformer_heads = 2 ** trial.suggest_int('num_transformer_heads_pow', low=0, high=2)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', low=1, high=4)

    # Encoder Head
    encoder_n_layers = trial.suggest_int('encoder_n_layers', 1, 4)
    encoder_dims = [2 ** trial.suggest_int(f'encoder_dim_{i}_pow', low=4, high=8) for i in range(encoder_n_layers)]
    encoder_layers = [MLPLayerConfig(out_dim=d, dropout=0.2) for d in encoder_dims]

    # Classification Head
    classification_n_layers = trial.suggest_int('classification_n_layers', 1, 4)
    classification_dims = [2 ** trial.suggest_int(f'classification_dim_{i}_pow', low=4, high=8) for i in
                           range(classification_n_layers)]
    classification_layers = [MLPLayerConfig(out_dim=d, dropout=0.2) for d in classification_dims]

    return dict(
        encoder_layers=encoder_layers,
        embedding_dim=embedding_dim,
        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,
        classification_layers=classification_layers,
    )
