import torch.nn as nn

from python.utils.net_utils import MLPLayerConfig, build_mlp_from_config


class TabularRNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_layers: list[MLPLayerConfig],
                 embedding_dim: int,
                 rnn_hidden_dim: int,
                 num_classes: int
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # MLP Encoder Block
        self.mlp_encoder = build_mlp_from_config(encoder_layers, input_dim, embedding_dim)

        # RNN Block
        self.rnn = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Classifier Head
        self.classifier = nn.Linear(rnn_hidden_dim, num_classes)

    def forward(self, x):
        x = self.mlp_encoder(x)
        _, hidden = self.rnn(x)

        # Since num_layers is 1
        # Squeeze to achieve shape of (batch_size, rnn_hidden_dim) instead of (num_layers, batch_size, rnn_hidden_dim)
        last_hidden_state = hidden.squeeze(0)

        logits = self.classifier(last_hidden_state)

        return logits
