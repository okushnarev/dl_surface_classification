from typing import List
import torch.nn as nn

from python.utils.net_utils import CNNLayerConfig, MLPLayerConfig, build_cnn_from_config


class CNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_steps: int,
                 cnn_configs: List[CNNLayerConfig],
                 mlp_configs: List[MLPLayerConfig],
                 num_classes: int):
        """
        A flexible 1D CNN framework built from configuration lists.

        Args:
            input_dim (int): Number of input features/channels.
            num_steps (int): Length of the input sequence.
            cnn_configs (List[CNNLayerConfig]): List of configs for CNN blocks.
            mlp_configs (List[MLPLayerConfig]): List of configs for MLP blocks.
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

        mlp_layers = []
        current_in_features = flattened_size

        for i, config in enumerate(mlp_configs):
            mlp_layers.append(nn.Linear(current_in_features, config.out_dim))
            mlp_layers.append(nn.ReLU())

            if config.dropout > 0:
                mlp_layers.append(nn.Dropout(p=config.dropout))

            # Update the feature count for the next layer
            current_in_features = config.out_dim

        # Add the final output layer
        mlp_layers.append(nn.Linear(current_in_features, num_classes))

        # The classifier starts with a Flatten layer
        self.classifier = nn.Sequential(nn.Flatten(), *mlp_layers)

    def forward(self, x):
        features = self.cnn_feature_extractor(x)
        output = self.classifier(features)
        return output
