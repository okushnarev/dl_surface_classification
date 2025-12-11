from mamba_ssm import Mamba2
from torch import nn
from torch import Tensor

from models.configs.mamba_config import MambaConfig
from python.utils.net_utils import MLPLayerConfig, build_mlp_from_config


class MambaClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int,
            encoder_layers: list[MLPLayerConfig],
            mamba_config: dict | MambaConfig,
            embedding_dim: int,
            output_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        if isinstance(mamba_config, dict):
            mamba_config = MambaConfig(**mamba_config)

        # MLP Encoder Block
        self.mlp_encoder = build_mlp_from_config(encoder_layers, input_dim, embedding_dim)

        self.mamba = Mamba2(
            d_model=embedding_dim,
            **mamba_config.model_dump(),
        )

        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp_encoder(x)
        x = self.mamba(x)
        output = self.classifier(x[:, -1, :])
        return output
