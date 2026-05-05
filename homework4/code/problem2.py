import numpy as np
from torch_geometric.nn.norm import LayerNorm
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import deeplay as dl


class MultiHeadAttentionLayer(dl.DeeplayModule):
    """Multi-head attention layer with masking."""

    def __init__(self, num_features, num_heads):
        """Initialize multi-head attention."""
        super().__init__()
        self.num_features, self.num_heads = num_features, num_heads
        self.head_dim = num_features // num_heads  # Must be integer

        self.Wq = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wk = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wv = dl.Layer(torch.nn.Linear, num_features, num_features)
        self.Wout = dl.Layer(torch.nn.Linear, num_features, num_features)

    def forward(self, in_sequence, batch_indices):
        """Apply the multi-head attention mechanism to the input sequence."""
        seq_len, embed_dim = in_sequence.shape
        Q = self.Wq(in_sequence)
        Q = Q.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = self.Wk(in_sequence)
        K = K.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = self.Wv(in_sequence)
        V = V.view(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        attn_mask = torch.eq(batch_indices.unsqueeze(1), batch_indices.unsqueeze(0))
        attn_mask = attn_mask.unsqueeze(0)
        attn_scores = attn_scores.masked_fill(attn_mask == False, float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(1, 0, 2).contiguous()
        attn_output = attn_output.view(seq_len, self.num_features)
        return self.Wout(attn_output)


class TransformerEncoderLayer(dl.DeeplayModule):
    """Transformer encoder layer."""

    def __init__(self, num_features, num_heads, feedforward_dim, dropout=0.0):
        """Initialize transformer encoder layer."""
        super().__init__()

        self.self_attn = MultiHeadAttentionLayer(num_features, num_heads)
        self.attn_dropout = dl.Layer(torch.nn.Dropout, dropout)
        self.attn_skip = dl.Add()
        self.attn_norm = dl.Layer(LayerNorm, num_features, eps=1e-6)

        self.feedforward = dl.Sequential(
            dl.Layer(torch.nn.Linear, num_features, feedforward_dim),
            dl.Layer(torch.nn.ReLU),
            dl.Layer(torch.nn.Linear, feedforward_dim, num_features),
        )
        self.feedforward_dropout = dl.Layer(torch.nn.Dropout, dropout)
        self.feedforward_skip = dl.Add()
        self.feedforward_norm = dl.Layer(LayerNorm, num_features, eps=1e-6)

    def forward(self, in_sequence, batch_indices):
        """Refine sequence via attention and feedforward layers."""
        attns = self.self_attn(in_sequence, batch_indices)
        attns = self.attn_dropout(attns)
        attns = self.attn_skip(in_sequence, attns)
        attns = self.attn_norm(attns, batch_indices)

        out_sequence = self.feedforward(attns)
        out_sequence = self.feedforward_dropout(out_sequence)
        out_sequence = self.feedforward_skip(attns, out_sequence)
        out_sequence = self.feedforward_norm(out_sequence, batch_indices)

        return out_sequence


class TransformerEncoderModel(dl.DeeplayModule):
    """Transformer encoder model."""

    def __init__(
        self,
        vocab_size,
        num_features,
        num_heads,
        feedforward_dim,
        num_layers,
        out_dim,
        encoder=None,
        dropout=0.0,
    ):
        """Initialize transformer encoder model."""
        super().__init__()
        self.num_features = num_features
        self.embedding = dl.Layer(torch.nn.Embedding, vocab_size, num_features)

        if encoder:
            self.pos_encoder = encoder
            self.pos_encoder.dropout.configure(p=dropout)
        else:
            self.pos_encoder = torch.zeros(num_features)

        self.transformer_block = dl.LayerList()
        for _ in range(num_layers):
            self.transformer_block.append(
                TransformerEncoderLayer(
                    num_features,
                    num_heads,
                    feedforward_dim,
                    dropout=dropout,
                )
            )

        self.out_block = dl.Sequential(
            dl.Layer(torch.nn.Dropout, dropout),
            dl.Layer(torch.nn.Linear, num_features, num_features // 2),
            dl.Layer(torch.nn.ReLU),
            dl.Layer(torch.nn.Linear, num_features // 2, out_dim),
            dl.Layer(torch.nn.Sigmoid),
        )

    def forward(self, dict):
        """Predict sentiment of movie reviews."""
        in_sequence, batch_indices = dict["sequences"], dict["batch_indices"]

        embeddings = self.embedding(in_sequence) * self.num_features**0.5
        pos_embeddings = self.pos_encoder(embeddings, batch_indices)

        out_sequence = pos_embeddings
        for transformer_layer in self.transformer_block:
            out_sequence = transformer_layer(out_sequence, batch_indices)

        batch_size = torch.max(batch_indices) + 1
        aggregates = torch.zeros(
            batch_size, self.num_features, device=out_sequence.device
        )
        for batch_index in torch.unique(batch_indices):
            mask = batch_indices == batch_index
            aggregates[batch_index] = out_sequence[mask].mean(dim=0)

        pred_sentiment = self.out_block(aggregates).squeeze()
        return pred_sentiment


def plot_weather_data(data, columns):
    start, days, daily_samples = 0, 14, 144
    end = start + daily_samples * days

    fig, axs = plt.subplots(7, 2, figsize=(16, 12), sharex=True)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(np.arange(start, end), data[start:end, i], label=columns[i])
        ax.set_xlim(start, end)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.legend(fontsize=20)

        for day in range(1, days):
            ax.axvline(
                x=start + daily_samples * day,
                color="gray",
                linestyle="--",
                linewidth=0.5,
            )
    fig.tight_layout()
    fig.savefig("../figures/problem2/plot_weather_data.png")


def main():
    dataframe = pd.read_csv("jena_climate_2009_2016.csv", index_col=0)
    data = dataframe.values

    """
    header = dataframe.columns.tolist()
    plot_weather_data(data, header)
    """
    print(dataframe.head(10))


if __name__ == "__main__":
    main()
