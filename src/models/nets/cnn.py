import json
from pathlib import Path

import torch.nn as nn

from src.models.schemas import CNNLayerConfig, MLPLayerConfig, build_cnn_from_config, build_mlp_from_config


class CNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_steps: int,
                 cnn_configs: list[CNNLayerConfig],
                 mlp_configs: list[MLPLayerConfig],
                 num_classes: int):
        """
        A flexible 1D CNN framework built from configuration lists.

        Args:
            input_dim (int): Number of raw features/channels.
            num_steps (int): Length of the raw sequence.
            cnn_configs (list[CNNLayerConfig]): List of configs for CNN blocks.
            mlp_configs (list[MLPLayerConfig]): List of configs for MLP blocks.
            num_classes (int): Number of classes for the final output.
        """
        super().__init__()

        if not cnn_configs:
            raise ValueError("cnn_configs list cannot be empty.")

        # CNN part
        (self.cnn_feature_extractor,
         current_channels,
         current_num_steps) = build_cnn_from_config(cnn_configs, input_dim, num_steps)

        # MLP part
        if current_num_steps == 0:
            raise ValueError(
                "Sequence length became 0 after pooling. "
                "Reduce pooling layers or use a longer initial sequence."
            )

        flattened_size = current_channels * current_num_steps
        self.flatten = nn.Flatten()
        self.classifier = build_mlp_from_config(mlp_configs, flattened_size, num_classes)

    def forward(self, x):
        features = self.cnn_feature_extractor(x)
        flat_features = self.flatten(features)
        output = self.classifier(flat_features)
        return output


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        cnn_configs = [
            CNNLayerConfig(out_channels=config[f'cnn_out_ch_{idx}'], kernel_size=3)
            for idx in range(config['cnn_n_layers'])
        ]

        mlp_configs = [
            MLPLayerConfig(out_dim=config[f'mlp_dim_{idx}'], dropout=0.2)
            for idx in range(config['mlp_n_layers'])
        ]

        start_lr = config['lr']
    else:
        # Defaults
        cnn_configs = [
            CNNLayerConfig(out_channels=32, kernel_size=3),
            CNNLayerConfig(out_channels=64, kernel_size=3),
        ]
        mlp_configs = [
            MLPLayerConfig(out_dim=64, dropout=0.2)
        ]
        start_lr = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
            num_steps=sequence_length,
            cnn_configs=cnn_configs,
            mlp_configs=mlp_configs,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
        )
    )


def cnn_objective(trial, val_dataset, input_dim, num_steps, num_classes, batch_size, device, epochs, seed):
    """Defines a single trial using a fixed train/validation split."""

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    cnn_n_layers = trial.suggest_int('cnn_n_layers', 1, 3)
    cnn_channels = [2 ** trial.suggest_int(f'cnn_out_ch_{i}_pow', low=4, high=8) for i in range(cnn_n_layers)]
    mlp_n_layers = trial.suggest_int('mlp_n_layers', 1, 4)
    mlp_dims = [2 ** trial.suggest_int(f'mlp_dim_{i}_pow', low=4, high=8) for i in range(mlp_n_layers)]

    # Create DataLoaders for this trial
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    # Instantiate Model and Optimizer
    cnn_configs = [
        CNNLayerConfig(out_channels=ch, kernel_size=3)
        for ch in cnn_channels
    ]
    mlp_configs = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in mlp_dims
    ]
    model = CNNTrainWrapper(
        input_dim=input_dim,
        num_steps=num_steps,
        cnn_configs=cnn_configs,
        mlp_configs=mlp_configs,
        num_classes=num_classes
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
