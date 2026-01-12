import json
from pathlib import Path

import torch.nn as nn

from src.models.schemas import MLPLayerConfig, build_funnel_dims, build_mlp_from_config


class TabularRNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_layers: list[MLPLayerConfig],
                 embedding_dim: int,
                 rnn_hidden_dim: int,
                 num_classes: int
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # MLP Encoder Block
        self.mlp_encoder = build_mlp_from_config(encoder_layers, input_dim, embedding_dim)

        # RNN Block
        self.rnn = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Classifier Head
        self.classifier = nn.Linear(rnn_hidden_dim, num_classes)

    def forward(self, x):
        x = self.mlp_encoder(x)
        _, hidden = self.rnn(x)

        # Since num_layers is 1
        # Squeeze to achieve shape of (batch_size, rnn_hidden_dim) instead of (num_layers, batch_size, rnn_hidden_dim)
        last_hidden_state = hidden.squeeze(0)

        logits = self.classifier(last_hidden_state)

        return logits


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        dropout = config.get('dropout', 0.2)
        rnn_hidden_dim = config['rnn_hidden_dim']

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
        weight_decay = config['weight_decay']
    else:
        # Defaults
        embedding_dim = 32

        dropout = 0.2
        encoder_layers = [
            MLPLayerConfig(out_dim=64, dropout=dropout),
            MLPLayerConfig(out_dim=16, dropout=dropout),
        ]
        rnn_hidden_dim = 64
        start_lr = 1e-2
        weight_decay = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
            encoder_layers=encoder_layers,
            embedding_dim=embedding_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
            weight_decay=weight_decay,
        )
    )


def get_optuna_params(trial):
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    encoder_n_layers = trial.suggest_int('encoder_n_layers', 1, 4)
    encoder_initial_dim = 2 ** trial.suggest_int('encoder_initial_dim_pow', low=2, high=5)
    encoder_expand_factor = 2 ** trial.suggest_int('encoder_expand_factor_pow', low=0, high=2)
    encoder_dims = build_funnel_dims(encoder_initial_dim, encoder_n_layers, encoder_expand_factor)
    encoder_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in encoder_dims]

    # Embedding dim
    embedding_dim = encoder_dims[-1]

    rnn_hidden_dim = 2 ** trial.suggest_int('rnn_hidden_dim_pow', low=6, high=10)

    return dict(
        encoder_layers=encoder_layers,
        embedding_dim=embedding_dim,
        rnn_hidden_dim=rnn_hidden_dim,
    )
