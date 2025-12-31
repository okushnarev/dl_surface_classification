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


def build_funnel_dims(initial_dim: int, n_steps: int, factor: float = 1, silent=False) -> list[int]:
    """
    Generates a sequence of dimensions creating a funnel shape

    This utility is useful for defining architecture depths, such as the number
    of units in MLP layers or channel counts in CNN blocks. The sequence begins
    with ``initial_dim`` and applies the ``factor`` iteratively for ``n_steps``

    :param int initial_dim: The starting dimension size
    :param int n_steps: The total number of dimensions to generate, including
       the initial dimension
    :param float factor: The scaling factor applied at each step. Values < 1.0
       contract the dimensions, while values > 1.0 expand them.
       Defaults to 1
    :param bool silent: If ``True``, suppresses the ``ValueError`` when a
       dimension drops below 1, returning the partial list generated up to
       that point. Defaults to ``False``

    :return: A list of integers representing the calculated dimensions
    :rtype: list[int]

    :raises ValueError: If a calculated dimension becomes less than 1 and
       ``silent`` is ``False``

    :Example:

    >>> build_funnel_dims(100, 3, 0.5)
    [100, 50, 25]

    >>> build_funnel_dims(10, 3, 2.0)
    [10, 20, 40]
    """
    output = []

    current_dim: int = initial_dim
    for idx in range(n_steps):
        if current_dim < 1:
            if silent:
                return output
            else:
                raise ValueError(f'Cannot create dimension less than 1: {current_dim=}')
        output.append(current_dim)
        current_dim = int(current_dim * factor)

    return output


def build_cnn_from_config(configs: list[CNNLayerConfig], input_dim: int, num_steps: int) -> tuple[nn.Module, int, int]:
    structure = []

    current_channels = input_dim
    current_num_steps = num_steps

    for i, config in enumerate(configs):
        conv_layer = nn.Conv1d(
            in_channels=current_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            padding=config.padding
        )
        structure.append(conv_layer)
        structure.append(nn.ReLU())

        if config.use_pooling:
            structure.append(nn.MaxPool1d(kernel_size=config.pool_size))
            # Update the sequence length tracker
            current_num_steps //= config.pool_size

        # Update the channel count for the next layer
        current_channels = config.out_channels

    return nn.Sequential(*structure), current_channels, current_num_steps


def build_mlp_from_config(configs: list[MLPLayerConfig], input_dim: int, output_dim: int) -> nn.Module:
    structure = []
    current_dim = input_dim

    for i, config in enumerate(configs):
        structure.append(nn.Linear(current_dim, config.out_dim))
        structure.append(nn.ReLU())

        if config.dropout > 0:
            structure.append(nn.Dropout(p=config.dropout))

        # Update the feature count for the next layer
        current_dim = config.out_dim

    structure.append(nn.Linear(current_dim, output_dim))

    return nn.Sequential(*structure)
