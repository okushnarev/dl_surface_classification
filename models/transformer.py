import torch
import torch.nn as nn


def build_mlp(initial_dim: int, layers: list[int], output_dim: int, dropout: float = 0.2) -> nn.Module:
    current_dim = initial_dim
    structure = []
    for layer in layers:
        structure.append(nn.Linear(current_dim, layer))
        structure.append(nn.ReLU())
        structure.append(nn.Dropout(dropout))
        current_dim = layer

    structure.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*structure)


class Transformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_layers: list[int],  # TODO: Update to MLPLayerConfig
                 embedding_dim: int,
                 num_heads: int,
                 num_transformer_layers: int,
                 classification_layers: list[int],  # TODO: Update to MLPLayerConfig
                 num_classes: int,
                 device: str = 'cpu'):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.device = device

        # Encoder
        self.encoder = build_mlp(input_dim, encoder_layers, embedding_dim, dropout=0.2)

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
        self.classifier = build_mlp(embedding_dim, classification_layers, num_classes, dropout=0.2)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.encoder(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = self.classifier(x[:, 0, :])
        return x
