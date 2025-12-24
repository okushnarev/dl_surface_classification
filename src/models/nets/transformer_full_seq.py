import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.modules.individual_tokenizer import NumericalFeatureTokenizer
from src.models.schemas import MLPLayerConfig, build_mlp_from_config


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

        # Pass processed through transformer and classifier
        x = self.transformer(x)
        x = self.classifier(x[:, 0, :])

        return x


def prep_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']
        num_transformer_heads = config['num_transformer_heads']
        num_transformer_layers = config['num_transformer_layers']

        classification_layers = [
            MLPLayerConfig(out_dim=config[f'classification_dim_{idx}'], dropout=0.2)
            for idx in range(config['classification_n_layers'])
        ]

        start_lr = config['lr']
    else:
        # Defaults
        embedding_dim = 32
        num_transformer_heads = 1
        num_transformer_layers = 1

        classification_layers = [
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        start_lr = 1e-2

    return dict(
        model=dict(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers,
            classification_layers=classification_layers,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
        )
    )


def transformer_full_seq_objective(trial, val_dataset, input_dim, num_steps, num_classes, batch_size, device, epochs,
                                   seed):
    """Defines a single trial using a fixed train/validation split."""

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2)
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=2, high=7)
    num_transformer_heads = 2 ** trial.suggest_int('num_transformer_heads_pow', low=0, high=2)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', low=1, high=4)

    classification_n_layers = trial.suggest_int('classification_n_layers', 1, 4)
    classification_dims = [2 ** trial.suggest_int(f'classification_dim_{i}_pow', low=4, high=8) for i in
                           range(classification_n_layers)]
    classification_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in classification_dims
    ]

    # Create DataLoaders for this trial
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    # Instantiate Model and Optimizer
    model = TransformerFullSeq(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,
        classification_layers=classification_layers,
        num_classes=num_classes,
    ).to(device)

    accuracy = run_training_loop(
        trial,
        model,
        val_loader,
        epochs,
        lr,
        device
    )

    return accuracy
