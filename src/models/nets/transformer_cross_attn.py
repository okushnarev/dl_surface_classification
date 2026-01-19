import json
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel

from src.models.schemas import MLPLayerConfig, build_funnel_dims, build_mlp_from_config
from src.modules.cross_attention import CrossAttention, CrossAttentionLayer


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


class TransformerCrossAttn(nn.Module):
    def __init__(self, config: TransformerCrossAttnConfig | None = None, **kwargs):
        super().__init__()

        if config is None:
            config = TransformerCrossAttnConfig(**kwargs)

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
            embed_dim=self.embedding_dim,
            num_heads=config.num_cross_attn_heads,
            ff_config=config.cross_attn_ffn_config,
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


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        num_transformer_heads = config['num_transformer_heads']
        num_transformer_layers = config['num_transformer_layers']

        num_cross_attn_heads = config['num_cross_attn_heads']
        num_cross_attn_layers = config['num_cross_attn_layers']

        dropout = config.get('dropout', 0.2)
        # Encoder
        encoder_n_layers = config['encoder_n_layers']
        if 'encoder_initial_dim' in config:
            # New funnel approach
            encoder_initial_dim = config['encoder_initial_dim']
            encoder_expand_factor = config['encoder_expand_factor']

            encoder_dims = build_funnel_dims(encoder_initial_dim, encoder_n_layers, encoder_expand_factor, silent=True)
            embedding_dim = encoder_dims[-1]
        else:
            # Backward compatibility
            encoder_dims = [config[f'encoder_dim_{idx}'] for idx in range(encoder_n_layers)]
            embedding_dim = config['embedding_dim']

        encoder_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in encoder_dims]

        # Classifier
        classification_n_layers = config['classification_n_layers']
        if 'classification_initial_dim' in config:
            # New funnel approach
            classification_initial_dim = config['classification_initial_dim']
            classification_expand_factor = config['classification_expand_factor']

            classification_dims = build_funnel_dims(classification_initial_dim, classification_n_layers,
                                                    classification_expand_factor)
        else:
            # Backward compatibility
            classification_dims = [config[f'classification_dim_{idx}'] for idx in range(classification_n_layers)]

        classification_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in classification_dims]

        # Cross-Attention FFN
        cross_attn_n_layers = config['cross_attn_n_layers']
        if 'cross_attn_initial_dim' in config:
            # New funnel approach
            cross_attn_initial_dim = config['cross_attn_initial_dim']
            cross_attn_expand_factor = config['cross_attn_expand_factor']

            cross_attn_dims = build_funnel_dims(cross_attn_initial_dim, cross_attn_n_layers,
                                                    cross_attn_expand_factor)
        else:
            # Backward compatibility
            cross_attn_dims = [config[f'cross_attn_dim_{idx}'] for idx in range(cross_attn_n_layers)]

        cross_attn_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in cross_attn_dims]

        start_lr = config['lr']
        weight_decay = config.get('weight_decay', 1e-2)

    else:
        # Defaults
        embedding_dim = 32
        num_transformer_heads = 1
        num_transformer_layers = 1

        num_cross_attn_heads = 1
        num_cross_attn_layers = 1

        dropout = 0.2
        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=dropout),
            MLPLayerConfig(out_dim=32, dropout=dropout),
        ]

        classification_layers = [
            MLPLayerConfig(out_dim=32, dropout=dropout),
        ]

        cross_attn_layers = [
            MLPLayerConfig(out_dim=embedding_dim * 2, dropout=dropout),
        ]

        start_lr = 1e-2
        weight_decay = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,

            encoder_layers=encoder_layers,

            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers,

            num_cross_attn_heads=num_cross_attn_heads,
            num_cross_attn_layers=num_cross_attn_layers,
            cross_attn_ffn_config=cross_attn_layers,

            classification_layers=classification_layers,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
            weight_decay=weight_decay,
        )
    )


def get_optuna_params(trial):
    dropout = 0.2

    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=4, high=8)

    num_transformer_heads = 2 ** trial.suggest_int('num_transformer_heads_pow', low=0, high=2)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', low=1, high=4)

    num_cross_attn_heads = 2 ** trial.suggest_int('num_cross_attn_heads_pow', low=0, high=2)
    num_cross_attn_layers = trial.suggest_int('num_cross_attn_layers', low=1, high=4)

    # Encoder Head
    encoder_n_layers = trial.suggest_int('encoder_n_layers', 1, 4)
    encoder_initial_dim = 2 ** trial.suggest_int('encoder_initial_dim_pow', low=2, high=5)
    encoder_expand_factor = 2 ** trial.suggest_int('encoder_expand_factor_pow', low=0, high=2)
    encoder_dims = build_funnel_dims(encoder_initial_dim, encoder_n_layers, encoder_expand_factor)
    encoder_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in encoder_dims]

    # Embedding dim
    embedding_dim = encoder_dims[-1]

    # Classification Layers
    classification_n_layers = trial.suggest_int('classification_n_layers', 1, 4)
    classification_initial_dim = 2 ** trial.suggest_int('classification_initial_dim_pow', low=6, high=10)
    classification_expand_factor = 2 ** trial.suggest_int('classification_expand_factor_pow', low=-2, high=0)
    classification_dims = build_funnel_dims(classification_initial_dim, classification_n_layers,
                                            classification_expand_factor, silent=True)
    classification_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in classification_dims]

    # Cross Attn FFN
    cross_attn_n_layers = trial.suggest_int('cross_attn_n_layers', 1, 4)
    cross_attn_initial_dim = 2 ** trial.suggest_int('cross_attn_initial_dim_pow', low=5, high=10)
    cross_attn_expand_factor = 2 ** trial.suggest_int('cross_attn_expand_factor_pow', low=0, high=0)
    cross_attn_dims = build_funnel_dims(cross_attn_initial_dim, cross_attn_n_layers, cross_attn_expand_factor)
    cross_attn_layers = [MLPLayerConfig(out_dim=d, dropout=dropout) for d in cross_attn_dims]

    return dict(
        embedding_dim=embedding_dim,
        encoder_layers=encoder_layers,
        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,
        num_cross_attn_heads=num_cross_attn_heads,
        num_cross_attn_layers=num_cross_attn_layers,
        cross_attn_ffn_config=cross_attn_layers,
        classification_layers=classification_layers,
    )
