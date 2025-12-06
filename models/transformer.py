import torch
import torch.nn as nn

from python.utils.net_utils import MLPLayerConfig, build_mlp_from_config


class Transformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_layers: list[MLPLayerConfig],
                 embedding_dim: int,
                 num_heads: int,
                 num_transformer_layers: int,
                 classification_layers: list[MLPLayerConfig],
                 num_classes: int):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = build_mlp_from_config(encoder_layers, input_dim, embedding_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_transformer_layers
        )

        # Classification head
        self.classifier = build_mlp_from_config(classification_layers, embedding_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.encoder(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = self.classifier(x[:, 0, :])
        return x
