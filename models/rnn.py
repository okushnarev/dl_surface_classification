import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset


class TabularRNN(nn.Module):
    def __init__(self, input_dim, mlp_hidden_dims, embedding_dim, rnn_hidden_dim, num_classes, device='cpu'):
        super(TabularRNN, self).__init__()
        self.embedding_dim = embedding_dim

        # --- Build MLP Encoder Block ---
        # This MLP will process each row (time step) of the sequence individually.
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

        # --- RNN (GRU) Block ---
        self.rnn = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # --- Classifier Head ---
        self.classifier = nn.Linear(rnn_hidden_dim, num_classes)
        self.to(device)

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # 1. Reshape for MLP: Apply MLP to each time step in the sequence
        # (batch_size, seq_len, input_dim) -> (batch_size * seq_len, input_dim)
        x_reshaped = x.view(batch_size * seq_len, -1)

        embedding_reshaped = self.mlp_encoder(x_reshaped)

        # 2. Reshape back for RNN
        # (batch_size * seq_len, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        rnn_input = embedding_reshaped.view(batch_size, seq_len, self.embedding_dim)

        # 3. Pass the sequence of embeddings through the GRU
        # PyTorch automatically handles the hidden state initialization.
        # rnn_output shape: (batch_size, seq_len, rnn_hidden_dim)
        # hidden shape: (num_layers, batch_size, rnn_hidden_dim)
        _, hidden = self.rnn(rnn_input)

        # 4. Get the last hidden state
        # We squeeze out the num_layers dimension (as it's 1)
        # Shape becomes: (batch_size, rnn_hidden_dim)
        last_hidden_state = hidden.squeeze(0)

        # 5. Pass through the final classifier
        logits = self.classifier(last_hidden_state)

        return logits
