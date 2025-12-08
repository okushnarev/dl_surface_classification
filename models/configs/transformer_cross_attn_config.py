from pydantic import BaseModel

from python.utils.net_utils import MLPLayerConfig


class TransformerCrossAttnConfig(BaseModel):
    input_dim: int
    sequence_length: int
    embedding_dim: int

    encoder_layers: list[MLPLayerConfig]

    num_transformer_heads: int
    num_transformer_layers: int

    num_cross_attn_heads: int
    num_cross_attn_layers: int
    cross_attn_ffn_config: list[MLPLayerConfig]

    classification_layers: list[MLPLayerConfig]
    num_classes: int
