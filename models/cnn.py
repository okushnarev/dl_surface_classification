import torch.nn as nn

from python.utils.net_utils import CNNLayerConfig, MLPLayerConfig, build_cnn_from_config, build_mlp_from_config


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
            input_dim (int): Number of input features/channels.
            num_steps (int): Length of the input sequence.
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
