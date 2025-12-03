import json

from models.transformer import Transformer
from python.train_seq import train

def load_mlp_layers(config, name):
    layers = []
    for idx in range(config[f'{name}_n_layers']):
        layers.append(config[f'{name}_dim_{idx}'])

    return layers

def prep_transformer_cfg(cfg_path, input_dim, num_classes, sequence_length):
    if cfg_path is not None:
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']
        num_heads = config['num_heads']
        num_transformer_layers = config['num_transformer_layers']

        encoder_layers = load_mlp_layers(config, 'encoder')
        classification_layers = load_mlp_layers(config, 'classification')

        start_lr = config['lr']
    else:
        encoder_layers = [16, 32]
        embedding_dim = 32
        num_heads = 2
        num_transformer_layers = 2
        classification_layers = [32]

        start_lr = 1e-2

    cfg = dict(
        model=dict(
            input_dim=input_dim,
            encoder_layers=encoder_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
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
    train(Transformer, prep_transformer_cfg)


if __name__ == '__main__':
    main()
