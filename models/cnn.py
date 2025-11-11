from typing import List
from pydantic import BaseModel
import torch.nn as nn


class CNNLayerConfig(BaseModel):
    """Configuration for a single 1D Convolutional block."""
    out_channels: int
    kernel_size: int
    padding: str = 'same'
    use_pooling: bool = True
    pool_size: int = 2


class MLPLayerConfig(BaseModel):
    """Configuration for a single MLP block."""
    out_dim: int
    dropout: float = 0.0


class CNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_steps: int,
                 cnn_configs: List[CNNLayerConfig],
                 mlp_configs: List[MLPLayerConfig],
                 output_dim: int):
        """
        A flexible 1D CNN framework built from configuration lists.

        Args:
            input_dim (int): Number of input features/channels.
            num_steps (int): Length of the input sequence.
            cnn_configs (List[CNNLayerConfig]): List of configs for CNN blocks.
            mlp_configs (List[MLPLayerConfig]): List of configs for MLP blocks.
            output_dim (int): Number of classes for the final output.
        """
        super().__init__()

        if not cnn_configs:
            raise ValueError("cnn_configs list cannot be empty.")

        # CNN part
        cnn_layers = []
        current_channels = input_dim
        current_num_steps = num_steps

        for i, config in enumerate(cnn_configs):
            conv_layer = nn.Conv1d(
                in_channels=current_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                padding=config.padding
            )
            cnn_layers.append(conv_layer)
            cnn_layers.append(nn.ReLU())

            if config.use_pooling:
                cnn_layers.append(nn.MaxPool1d(kernel_size=config.pool_size))
                # Update the sequence length tracker
                current_num_steps //= config.pool_size

            # Update the channel count for the next layer
            current_channels = config.out_channels

        self.cnn_feature_extractor = nn.Sequential(*cnn_layers)

        # MLP part
        if current_num_steps == 0:
            raise ValueError(
                "Sequence length became 0 after pooling. "
                "Reduce pooling layers or use a longer initial sequence."
            )

        flattened_size = current_channels * current_num_steps

        mlp_layers = []
        current_in_features = flattened_size

        for i, config in enumerate(mlp_configs):
            mlp_layers.append(nn.Linear(current_in_features, config.out_dim))
            mlp_layers.append(nn.ReLU())

            if config.dropout_rate > 0:
                mlp_layers.append(nn.Dropout(p=config.dropout_rate))

            # Update the feature count for the next layer
            current_in_features = config.out_dim

        # Add the final output layer
        mlp_layers.append(nn.Linear(current_in_features, output_dim))

        # The classifier starts with a Flatten layer
        self.classifier = nn.Sequential(nn.Flatten(), *mlp_layers)

    def forward(self, x):
        features = self.cnn_feature_extractor(x)
        output = self.classifier(features)
        return output
