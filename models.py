import torch
import torch.nn as nn
from utils import Time2Vec

torch.manual_seed(0)


class AttentionLSTM(nn.Module):
    """
    Multihead-attention model + LSTM layer.
    """
    def __init__(self, embed_dim, out_size, hidden_size=17, n_layers=2):
        super(AttentionLSTM, self).__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=embed_dim
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out, weights = self.att(X, X, X)
        out, (h, c) = self.lstm(out)
        out = self.lin(out)
        return out


class VanillaLSTM(nn.Module):
    """
    Multihead-attention model + LSTM layer.
    """
    def __init__(self, input_size, out_size, hidden_size=20, n_layers=2):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out, (h, c) = self.lstm(X)
        out = self.lin(out)
        return out


class EmbeddingLSTM(nn.Module):
    """
    Time2vec embedding + LSTM.
    """
    def __init__(
        self,
        linear_channel,
        period_channel,
        input_channel,
        input_size,
        out_size,
        hidden_size=19,
        n_layers=2
    ):
        super(EmbeddingLSTM, self).__init__()
        self.emb = Time2Vec(linear_channel, period_channel, input_channel)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out = self.emb(X)
        out, (h, c) = self.lstm(out)
        out = self.lin(out)
        return out


class AttentionEmbeddingLSTM(nn.Module):
    """
    Time2vec embedding + Attention + LSTM.
    """
    def __init__(
        self,
        linear_channel,
        period_channel,
        input_channel,
        input_size,
        out_size,
        hidden_size=16,
        n_layers=2
    ):
        super(AttentionEmbeddingLSTM, self).__init__()

        self.emb = Time2Vec(linear_channel, period_channel, input_channel)
        self.att = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=input_size
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out = self.emb(X)
        out, w = self.att(out, out, out)
        out, (h, c) = self.lstm(out)
        out = self.lin(out)
        return out
