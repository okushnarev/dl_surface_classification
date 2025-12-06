import torch.nn as nn


class TabularRNN(nn.Module):
    def __init__(self,
                 input_dim,
                 mlp_hidden_dims,  # TODO: update to MLPLayerConfig
                 embedding_dim,
                 rnn_hidden_dim,
                 num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim

        # MLP Encoder Block
        mlp_layers = []
        current_features = input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers += [
                nn.Linear(current_features, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ]
            current_features = hidden_dim
        mlp_layers.append(nn.Linear(current_features, self.embedding_dim))
        self.mlp_encoder = nn.Sequential(*mlp_layers)

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
