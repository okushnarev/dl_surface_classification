import copy

import torch
from torch import nn as nn

from src.models.schemas import MLPLayerConfig, build_mlp_from_config


def _get_clones(module, N):
    # Copy of protected torch method from torch.nn.modules.transformer
    # Copied for code clarity
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CrossAttentionLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            ff_config: list[MLPLayerConfig],
            dropout: float = 0.0,
            batch_first: bool = True):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.ff = build_mlp_from_config(ff_config, embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
        )
        # Add and Norm
        query = self.norm1(query + self.dropout(attn_output))
        # FFN
        ff_output = self.ff(query)
        # Add and Norm
        output = self.norm2(query + self.dropout(ff_output))

        return output


class CrossAttention(nn.Module):
    def __init__(self, cross_attn_layer: CrossAttentionLayer, num_layers: int):
        super().__init__()
        self.layers = _get_clones(cross_attn_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        output = query
        for layer in self.layers:
            output = layer(output, key_value)
        # LayerNorm is embedded in CrossAttentionLayer
        return output
