import json
from pathlib import Path

import torch.nn as nn

from src.models.schemas import CNNLayerConfig, MLPLayerConfig, build_cnn_from_config, build_funnel_dims, \
    build_mlp_from_config


class CNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 sequence_length: int,
                 cnn_configs: list[CNNLayerConfig],
                 mlp_configs: list[MLPLayerConfig],
                 num_classes: int):
        """
        A flexible 1D CNN framework built from configuration lists

        Args:
            input_dim (int): Number of raw features/channels
            sequence_length (int): Length of the raw sequence
            cnn_configs (list[CNNLayerConfig]): List of configs for CNN blocks
            mlp_configs (list[MLPLayerConfig]): List of configs for MLP blocks
            num_classes (int): Number of classes for the final output
        """
        super().__init__()

        if not cnn_configs:
            raise ValueError('cnn_configs list cannot be empty.')

        # CNN part
        (self.cnn_feature_extractor,
         current_channels,
         current_num_steps) = build_cnn_from_config(cnn_configs, input_dim, sequence_length)

        # MLP part
        if current_num_steps == 0:
            raise ValueError(
                'Sequence length became 0 after pooling. '
                'Reduce pooling layers or use a longer initial sequence.'
            )

        flattened_size = current_channels * current_num_steps
        self.flatten = nn.Flatten()
        self.classifier = build_mlp_from_config(mlp_configs, flattened_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.cnn_feature_extractor(x)
        flat_features = self.flatten(features)
        output = self.classifier(flat_features)
        return output


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        cnn_n_layers = config['cnn_n_layers']
        if 'cnn_initial_dim' in config:
            # New funnel approach
            cnn_initial_dim = config['cnn_initial_dim']
            cnn_expand_factor = config['cnn_expand_factor']

            cnn_dims = build_funnel_dims(cnn_initial_dim, cnn_n_layers, cnn_expand_factor)
        else:
            # Backward compatibility
            cnn_dims = [config[f'cnn_out_ch_{idx}'] for idx in range(cnn_n_layers)]

        cnn_configs = [CNNLayerConfig(out_channels=d, kernel_size=3) for d in cnn_dims]

        dropout = config.get('dropout', 0.2)
        mlp_n_layers = config['mlp_n_layers']
        if 'mlp_initial_dim' in config:
            # New funnel approach
            mlp_initial_dim = config['mlp_initial_dim']
            mlp_expand_factor = config['mlp_expand_factor']

            mlp_dims = build_funnel_dims(mlp_initial_dim, mlp_n_layers, mlp_expand_factor, silent=True)
        else:
            # Backward compatibility
            mlp_dims = [config[f'mlp_dim_{idx}'] for idx in range(mlp_n_layers)]

        mlp_configs = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in mlp_dims]

        start_lr = config['lr']
        weight_decay = config.get('weight_decay', 1e-4)

    else:
        # Defaults
        cnn_configs = [
            CNNLayerConfig(out_channels=32, kernel_size=3),
            CNNLayerConfig(out_channels=64, kernel_size=3),
        ]

        dropout = 0.2
        mlp_configs = [
            MLPLayerConfig(out_dim=64, dropout=dropout),
        ]
        start_lr = 1e-2
        weight_decay = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
            sequence_length=sequence_length,
            cnn_configs=cnn_configs,
            mlp_configs=mlp_configs,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
            weight_decay=weight_decay,
        )
    )


def get_optuna_params(trial):
    dropout = 0.2

    cnn_n_layers = trial.suggest_int('cnn_n_layers', 1, 3)
    cnn_initial_dim = 2 ** trial.suggest_int('cnn_initial_dim_pow', low=2, high=5)
    cnn_expand_factor = 2 ** trial.suggest_int('cnn_expand_factor_pow', low=0, high=2)
    cnn_channels = build_funnel_dims(cnn_initial_dim, cnn_n_layers, cnn_expand_factor)
    cnn_configs = [CNNLayerConfig(out_channels=ch, kernel_size=3) for ch in cnn_channels]

    mlp_n_layers = trial.suggest_int('mlp_n_layers', 1, 4)
    mlp_initial_dim = 2 ** trial.suggest_int('mlp_initial_dim_pow', low=6, high=10)
    mlp_expand_factor = 2 ** trial.suggest_int('mlp_expand_factor_pow', low=-2, high=0)
    mlp_dims = build_funnel_dims(mlp_initial_dim, mlp_n_layers, mlp_expand_factor, silent=True)

    mlp_configs = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in mlp_dims]

    return dict(
        cnn_configs=cnn_configs,
        mlp_configs=mlp_configs
    )
