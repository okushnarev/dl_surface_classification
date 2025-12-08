import torch
import torch.nn as nn

from models.configs.transformer_cross_attn_config import TransformerCrossAttnConfig
from models.utils.cross_attn import CrossAttention, CrossAttentionLayer
from python.utils.net_utils import build_mlp_from_config


class TransformerCrossAttn(nn.Module):
    def __init__(self, config: TransformerCrossAttnConfig):
        super().__init__()

        self.input_dim = config.input_dim
        self.embedding_dim = config.embedding_dim
        self.sequence_length = config.sequence_length
        self.num_classes = config.num_classes

        # Encoder
        self.encoder_in = build_mlp_from_config(config.encoder_layers, self.input_dim, self.embedding_dim)
        self.encoder_seq = build_mlp_from_config(config.encoder_layers, self.sequence_length, self.embedding_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=config.num_transformer_heads,
            batch_first=True,
            dropout=0.2
        )
        self.transformer_in = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.num_transformer_layers
        )
        self.transformer_cross = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.num_transformer_layers
        )

        cross_attn_layer = CrossAttentionLayer(
            self.embedding_dim,
            config.num_cross_attn_heads,
            config.cross_attn_ffn_config,
            dropout=0.2
        )
        self.cross_attn = CrossAttention(
            cross_attn_layer,
            num_layers=config.num_cross_attn_layers
        )

        # Classification head
        self.classifier = build_mlp_from_config(config.classification_layers, self.embedding_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Encode main sequence with CLS token
        x_in = self.encoder_in(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_in = torch.cat((cls_tokens, x_in), dim=1)
        x_in = self.transformer_in(x_in)

        # Encode 'transposed' sequence
        x_seq = self.encoder_seq(x.permute(0, 2, 1))
        x_seq = self.transformer_cross(x_seq)

        # Perform cross-attention
        x = self.cross_attn(x_in, x_seq)

        # Prepare output
        x = self.classifier(x[:, 0, :])

        return x
