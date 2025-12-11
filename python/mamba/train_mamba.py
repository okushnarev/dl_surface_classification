import json
from pathlib import Path

from models.configs.mamba_config import MambaConfig
from models.mamba import MambaClassifier
from python.train_seq import train
from python.utils.net_utils import MLPLayerConfig


def prep_mamba_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int = None):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']

        # Mamba params
        d_state = config['d_state']
        headdim = config['headdim']
        mamba_config = MambaConfig(
            d_state=d_state,
            headdim=headdim,
        )

        encoder_layers = [
            MLPLayerConfig(out_dim=config[f'encoder_dim_{idx}'], dropout=0.2)
            for idx in range(config['encoder_n_layers'])
        ]

        start_lr = config['lr']
    else:
        embedding_dim = 32

        encoder_layers = [
            MLPLayerConfig(out_dim=16, dropout=0.2),
            MLPLayerConfig(out_dim=32, dropout=0.2),
        ]

        mamba_config = MambaConfig()

        start_lr = 1e-2

    cfg = dict(
        model=dict(
            input_dim=input_dim,
            encoder_layers=encoder_layers,
            mamba_config=mamba_config,
            embedding_dim=embedding_dim,
            output_dim=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
        )
    )

    return cfg


def main():
    train(MambaClassifier, prep_mamba_cfg)


if __name__ == '__main__':
    main()
