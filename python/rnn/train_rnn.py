import json

from models.rnn import TabularRNN
from python.train_seq import train


def prep_rnn_cfg(cfg_path, input_dim, num_classes):
    if cfg_path is not None:
        with open(cfg_path, 'r') as f:
            config = json.load(f)['params']

        embedding_dim = config['embedding_dim']
        rnn_hidden_dim = config['rnn_hidden_dim']
        mlp_hidden_dims = []
        for idx in range(config['mlp_n_layers']):
            mlp_hidden_dims.append(config[f'mlp_dim_{idx}'])

        start_lr = config['lr']
    else:
        embedding_dim = 32
        mlp_hidden_dims = [64, 16]
        rnn_hidden_dim = 64
        start_lr = 1e-2

    cfg = dict(
        model=dict(
            input_dim=input_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            embedding_dim=embedding_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            num_classes=num_classes,
        ),
        optimizer=dict(
            start_lr=start_lr,
        )
    )

    return cfg


def main():
    train(TabularRNN, prep_rnn_cfg)


if __name__ == '__main__':
    main()
