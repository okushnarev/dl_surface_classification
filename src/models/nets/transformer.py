import json
from pathlib import Path

import torch
import torch.nn as nn

from src.models.schemas import MLPLayerConfig, build_funnel_dims, build_mlp_from_config
from src.modules.positional_encoder import PositionalEncoding


class Transformer(nn.Module):
    max_dim_size = 256
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

        # Positional encoder
        self.pos_encoder = PositionalEncoding(embedding_dim)

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
        self.init_transformer_weights()

        # Classification head
        self.classifier = build_mlp_from_config(classification_layers, embedding_dim, num_classes)

    def init_transformer_weights(self):
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.encoder(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.classifier(x[:, 0, :])
        return x


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        num_transformer_heads = config.get('num_transformer_heads', 1)
        num_transformer_layers = config['num_transformer_layers']

        max_dim_size = Transformer.max_dim_size
        dropout = config.get('dropout', 0.2)

        encoder_n_layers = config['encoder_n_layers']
        if 'encoder_initial_dim' in config:
            # New funnel approach
            encoder_initial_dim = config['encoder_initial_dim']
            encoder_expand_factor = config['encoder_expand_factor']

            encoder_dims = build_funnel_dims(encoder_initial_dim, encoder_n_layers, encoder_expand_factor, silent=True,
                                             top=max_dim_size)
            embedding_dim = encoder_dims[-1]
        else:
            # Backward compatibility
            encoder_dims = [config[f'encoder_dim_{idx}'] for idx in range(encoder_n_layers)]
            embedding_dim = config['embedding_dim']

        encoder_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in encoder_dims]

        classification_n_layers = config['classification_n_layers']
        if 'classification_initial_dim' in config:
            # New funnel approach
            classification_initial_dim = config['classification_initial_dim']
            classification_expand_factor = config['classification_expand_factor']

            classification_dims = build_funnel_dims(classification_initial_dim, classification_n_layers,
                                                    classification_expand_factor, top=max_dim_size)
        else:
            # Backward compatibility
            classification_dims = [config[f'classification_dim_{idx}'] for idx in range(classification_n_layers)]

        classification_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in classification_dims]

        start_lr = config['lr']
        weight_decay = config.get('weight_decay', 1e-2)
    else:
        # Defaluts
        embedding_dim = 32
        num_transformer_heads = 1
        num_transformer_layers = 1

        dropout = 0.2
        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=dropout),
            MLPLayerConfig(out_dim=32, dropout=dropout),
        ]

        classification_layers = [
            MLPLayerConfig(out_dim=32, dropout=dropout),
        ]

        start_lr = 1e-2
        weight_decay = 1e-2

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
            weight_decay=weight_decay,
        )
    )


def get_optuna_params(trial):
    dropout = 0.2
    max_dim_size = Transformer.max_dim_size

    num_transformer_heads = 1
    num_transformer_layers = trial.suggest_int('num_transformer_layers', low=1, high=2)

    # Encoder Head
    encoder_n_layers = trial.suggest_int('encoder_n_layers', 1, 2)
    encoder_initial_dim = 2 ** trial.suggest_int('encoder_initial_dim_pow', low=2, high=8)
    encoder_expand_factor = 2 ** trial.suggest_int('encoder_expand_factor_pow', low=0, high=2)
    encoder_dims = build_funnel_dims(encoder_initial_dim, encoder_n_layers, encoder_expand_factor, top=max_dim_size)
    encoder_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in encoder_dims]

    # Embedding dim
    embedding_dim = encoder_dims[-1]

    # Classification Head
    classification_n_layers = trial.suggest_int('classification_n_layers', 1, 2)
    classification_initial_dim = 2 ** trial.suggest_int('classification_initial_dim_pow', low=6, high=8)
    classification_expand_factor = 2 ** trial.suggest_int('classification_expand_factor_pow', low=-2, high=0)
    classification_dims = build_funnel_dims(classification_initial_dim, classification_n_layers,
                                            classification_expand_factor, silent=True, top=max_dim_size)
    classification_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in classification_dims]

    return dict(
        encoder_layers=encoder_layers,
        embedding_dim=embedding_dim,
        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,
        classification_layers=classification_layers,
    )
