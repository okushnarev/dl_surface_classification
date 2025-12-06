import json
from pathlib import Path

from models.cnn import CNN, CNNLayerConfig, MLPLayerConfig
from python.train_seq import train


def prep_cnn_cfg(cfg_path: Path, input_dim: int, num_classes: int, sequence_length: int):
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        cnn_configs = [
            CNNLayerConfig(out_channels=config[f'cnn_out_ch_{idx}'], kernel_size=3)
            for idx in range(config['cnn_n_layers'])
        ]

        mlp_configs = [
            MLPLayerConfig(out_dim=config[f'mlp_dim_{idx}'], dropout=0.2)
            for idx in range(config['mlp_n_layers'])
        ]

        start_lr = config['lr']
    else:
        cnn_configs = [
            CNNLayerConfig(out_channels=32, kernel_size=3),
            CNNLayerConfig(out_channels=64, kernel_size=3),
        ]
        mlp_configs = [
            MLPLayerConfig(out_dim=64, dropout=0.2)
        ]
        start_lr = 1e-2

    cfg = dict(
        model=dict(
            input_dim=input_dim,
            num_steps=sequence_length,
            cnn_configs=cnn_configs,
            mlp_configs=mlp_configs,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
        )
    )

    return cfg


class CNNTrainWrapper(CNN):
    def __init__(self,
                 input_dim: int,
                 num_steps: int,
                 cnn_configs: list[CNNLayerConfig],
                 mlp_configs: list[MLPLayerConfig],
                 num_classes: int):
        super().__init__(input_dim, num_steps, cnn_configs, mlp_configs, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return super().forward(x)


def main():
    train(CNNTrainWrapper, prep_cnn_cfg)


if __name__ == '__main__':
    main()
