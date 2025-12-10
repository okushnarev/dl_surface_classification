import json
from pathlib import Path

from models.transformer_full_seq import TransformerFullSeq
from python.train_seq import train
from python.utils.net_utils import MLPLayerConfig


def prep_transformer_full_seq_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
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
        embedding_dim = 32
        num_transformer_heads = 1
        num_transformer_layers = 1

        classification_layers = [
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        start_lr = 1e-2

    cfg = dict(
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

    return cfg


def main():
    train(TransformerFullSeq, prep_transformer_full_seq_cfg)


if __name__ == '__main__':
    main()
