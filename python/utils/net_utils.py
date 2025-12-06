import torch.nn as nn
from pydantic import BaseModel


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


def build_cnn_from_config(configs: list[CNNLayerConfig], input_dim: int, num_steps: int) -> tuple[nn.Module, int, int]:
    cnn_layers = []
    current_channels = input_dim
    current_num_steps = num_steps

    for i, config in enumerate(configs):
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

    return nn.Sequential(*cnn_layers), current_channels, current_num_steps
