import json
from pathlib import Path

from models.transformer_cross_attn import TransformerCrossAttn
from python.train_seq import train
from python.utils.net_utils import MLPLayerConfig


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


def main():
    train(TransformerCrossAttn, prep_transformer_cross_attn_cfg)


if __name__ == '__main__':
    main()
