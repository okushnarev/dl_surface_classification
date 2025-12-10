import torch
import torch.nn as nn

from models.utils.individual_tokenizer import NumericalFeatureTokenizer
from python.utils.net_utils import MLPLayerConfig, build_mlp_from_config


class TransformerFullSeq(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 classification_layers: list[MLPLayerConfig],
                 num_classes: int):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Encoder
        self.tokenizer = NumericalFeatureTokenizer(input_dim, embedding_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))

        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_transformer_heads,
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

        # Tokenize
        x = self.tokenizer(x)

        # Flatten sequence_len and input_dim
        x = x.view(batch_size, -1, self.embedding_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Pass data through transformer and classifier
        x = self.transformer(x)
        x = self.classifier(x[:, 0, :])

        return x
