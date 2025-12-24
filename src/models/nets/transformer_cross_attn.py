import json
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.data import DataLoader

from src.modules.cross_attention import CrossAttention, CrossAttentionLayer
from src.models.schemas import MLPLayerConfig, build_mlp_from_config


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


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']
        num_transformer_heads = config['num_transformer_heads']
        num_transformer_layers = config['num_transformer_layers']

        num_cross_attn_heads = config['num_cross_attn_heads']
        num_cross_attn_layers = config['num_cross_attn_layers']

        encoder_layers = [
            MLPLayerConfig(out_dim=config[f'encoder_dim_{idx}'], dropout=0.2)
            for idx in range(config['encoder_n_layers'])
        ]

        classification_layers = [
            MLPLayerConfig(out_dim=config[f'classification_dim_{idx}'], dropout=0.2)
            for idx in range(config['classification_n_layers'])
        ]

        cross_attn_layers = [
            MLPLayerConfig(out_dim=config[f'cross_attn_dim_{idx}'], dropout=0.2)
            for idx in range(config['cross_attn_n_layers'])
        ]

        start_lr = config['lr']
    else:
        # Defaults
        embedding_dim = 32
        num_transformer_heads = 1
        num_transformer_layers = 1

        num_cross_attn_heads = 1
        num_cross_attn_layers = 1

        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=0.2),
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        classification_layers = [
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        cross_attn_layers = [
            MLPLayerConfig(out_dim=embedding_dim * 2, dropout=0.2),
        ]

        start_lr = 1e-2

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
        )
    )


def transformer_cross_attn_objective(trial, val_dataset, input_dim, num_steps, num_classes, batch_size, device, epochs,
                                     seed):
    """Defines a single trial using a fixed train/validation split."""

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=4, high=8)

    num_transformer_heads = 2 ** trial.suggest_int('num_transformer_heads_pow', low=0, high=2)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', low=1, high=4)

    num_cross_attn_heads = 2 ** trial.suggest_int('num_cross_attn_heads_pow', low=0, high=2)
    num_cross_attn_layers = trial.suggest_int('num_cross_attn_layers', low=1, high=4)

    encoder_n_layers = trial.suggest_int('encoder_n_layers', 1, 4)
    encoder_dims = [2 ** trial.suggest_int(f'encoder_dim_{i}_pow', low=4, high=8) for i in range(encoder_n_layers)]

    classification_n_layers = trial.suggest_int('classification_n_layers', 1, 4)
    classification_dims = [2 ** trial.suggest_int(f'classification_dim_{i}_pow', low=4, high=8) for i in
                           range(classification_n_layers)]

    cross_attn_n_layers = trial.suggest_int('cross_attn_n_layers', 1, 4)
    cross_attn_dims = [2 ** trial.suggest_int(f'cross_attn_dim_{i}_pow', low=4, high=8) for i in
                       range(cross_attn_n_layers)]

    encoder_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in encoder_dims
    ]
    classification_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in classification_dims
    ]

    cross_attn_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in cross_attn_dims
    ]

    model_cfg = TransformerCrossAttnConfig(
        input_dim=input_dim,
        sequence_length=num_steps,
        embedding_dim=embedding_dim,

        encoder_layers=encoder_layers,

        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,

        num_cross_attn_heads=num_cross_attn_heads,
        num_cross_attn_layers=num_cross_attn_layers,
        cross_attn_ffn_config=cross_attn_layers,

        classification_layers=classification_layers,
        num_classes=num_classes,
    )

    # Instantiate Model and Optimizer
    model = TransformerCrossAttn(model_cfg).to(device)

    # Create DataLoaders for this trial
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    accuracy = run_training_loop(
        trial,
        model,
        val_loader,
        epochs,
        lr,
        device
    )

    return accuracy


def prep_transformer_cross_attn_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']
        num_transformer_heads = config['num_transformer_heads']
        num_transformer_layers = config['num_transformer_layers']

        num_cross_attn_heads = config['num_cross_attn_heads']
        num_cross_attn_layers = config['num_cross_attn_layers']

        encoder_layers = [
            MLPLayerConfig(out_dim=config[f'encoder_dim_{idx}'], dropout=0.2)
            for idx in range(config['encoder_n_layers'])
        ]

        classification_layers = [
            MLPLayerConfig(out_dim=config[f'classification_dim_{idx}'], dropout=0.2)
            for idx in range(config['classification_n_layers'])
        ]

        cross_attn_layers = [
            MLPLayerConfig(out_dim=config[f'cross_attn_dim_{idx}'], dropout=0.2)
            for idx in range(config['cross_attn_n_layers'])
        ]

        start_lr = config['lr']
    else:
        embedding_dim = 32
        num_transformer_heads = 1
        num_transformer_layers = 1

        num_cross_attn_heads = 1
        num_cross_attn_layers = 1

        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=0.2),
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        classification_layers = [
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        cross_attn_layers = [
            MLPLayerConfig(out_dim=embedding_dim * 2, dropout=0.2),
        ]

        start_lr = 1e-2

    cfg = dict(
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
        )
    )

    return cfg
