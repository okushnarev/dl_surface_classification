import json
from pathlib import Path

import torch
import torch.nn as nn

from src.models.schemas import MLPLayerConfig, build_funnel_dims, build_mlp_from_config
from src.modules.individual_tokenizer import NumericalFeatureTokenizer


class TransformerFullSeq(nn.Module):
    def __init__(self,
                 input_dim: int,
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
        self.tokenizer = NumericalFeatureTokenizer(input_dim, embedding_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))

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

        # Tokenize
        x = self.tokenizer(x)

        # Flatten sequence_len and input_dim
        x = x.view(batch_size, -1, self.embedding_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Pass processed through transformer and classifier
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

        dropout = config['dropout']
        classification_n_layers = config['classification_n_layers']
        if 'classification_initial_dim' in config:
            # New funnel approach
            classification_initial_dim = config['classification_initial_dim']
            classification_expand_factor = config['classification_expand_factor']

            classification_dims = build_funnel_dims(classification_initial_dim, classification_n_layers,
                                                    classification_expand_factor)
        else:
            # Backward compatibility
            classification_dims = [config[f'classification_dim_{idx}'] for idx in range(classification_n_layers)]

        classification_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in classification_dims]

        start_lr = config['lr']
        weight_decay = config['weight_decay']

    else:
        # Defaults
        embedding_dim = 32
        num_transformer_heads = 1
        num_transformer_layers = 1

        dropout = 0.2
        classification_layers = [
            MLPLayerConfig(out_dim=32, dropout=dropout),
        ]

        start_lr = 1e-2
        weight_decay = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
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
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=2, high=10)
    num_transformer_heads = 2 ** trial.suggest_int('num_transformer_heads_pow', low=0, high=2)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', low=1, high=4)

    # Classification Head
    classification_n_layers = trial.suggest_int('classification_n_layers', 1, 4)
    classification_initial_dim = 2 ** trial.suggest_int('classification_initial_dim_pow', low=6, high=10)
    classification_expand_factor = 2 ** trial.suggest_int('classification_expand_factor_pow', low=-2, high=0)
    classification_dims = build_funnel_dims(classification_initial_dim, classification_n_layers,
                                            classification_expand_factor, silent=True)
    classification_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in classification_dims]

    return dict(
        embedding_dim=embedding_dim,
        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,
        classification_layers=classification_layers,
    )
