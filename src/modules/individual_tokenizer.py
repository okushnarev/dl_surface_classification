import math

import torch
from torch import nn


class NumericalFeatureTokenizer(nn.Module):
    """
    Transforms continuous features to tokens / embeddings
    """

    def __init__(
            self,
            num_features: int,
            embedding_dim: int,
            bias: bool = True,
    ) -> None:

        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features, embedding_dim))
        self.bias = nn.Parameter(torch.Tensor(num_features, embedding_dim)) if bias else None

        # Init weights
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x
