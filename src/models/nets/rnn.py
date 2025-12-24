import json
from pathlib import Path

import torch.nn as nn

from src.models.schemas import MLPLayerConfig, build_mlp_from_config


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

        embedding_dim = config['embedding_dim']
        rnn_hidden_dim = config['rnn_hidden_dim']

        encoder_layers = [
            MLPLayerConfig(out_dim=config[f'mlp_dim_{idx}'], dropout=0.2)
            for idx in range(config['mlp_n_layers'])
        ]

        start_lr = config['lr']
    else:
        # Defaults
        embedding_dim = 32
        encoder_layers = [
            MLPLayerConfig(out_dim=64, dropout=0.2),
            MLPLayerConfig(out_dim=16, dropout=0.2),
        ]
        rnn_hidden_dim = 64
        start_lr = 1e-2

    return dict(
        model_kwargs=dict(
            input_dim=input_dim,
            encoder_layers=encoder_layers,
            embedding_dim=embedding_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            num_classes=num_classes,
        ),
        optimizer=dict(start_lr=start_lr)
    )



def get_rnn_optuna_params(trial):
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)

    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=2, high=8)
    rnn_hidden_dim = 2 ** trial.suggest_int('rnn_hidden_dim_pow', low=4, high=6)
    mlp_n_layers = trial.suggest_int('mlp_n_layers', 1, 4)

    mlp_dims = [2 ** trial.suggest_int(f'mlp_dim_{i}_pow', low=4, high=8) for i in range(mlp_n_layers)]

    encoder_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in mlp_dims
    ]

    return dict(
        model_kwargs=dict(
            encoder_layers=encoder_layers,
            embedding_dim=embedding_dim,
            rnn_hidden_dim=rnn_hidden_dim,
        ),
        lr=lr
    )
