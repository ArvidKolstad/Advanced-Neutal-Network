import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import deeplay as dl
import torch
from torch import nn


class GenerateAttentionMatrixes(nn.Module):
    def __init__(self, in_features, trans_dim, encoder=None):
        super().__init__()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = torch.zeros(trans_dim)
        self.embedder = nn.Linear(in_features, trans_dim)
        self.get_query = nn.Linear(trans_dim, trans_dim)
        self.get_key = nn.Linear(trans_dim, trans_dim)
        self.get_value = nn.Linear(trans_dim, trans_dim)

        self.get_attention_matrix = DotProductAttention()

    def forward(self, input_features):
        embedded = self.embedder(input_features) + self.encoder
        q = self.get_query(embedded)
        k = self.get_key(embedded)
        v = self.get_value(embedded)
        attention_matrix = self.get_attention_matrix(q, k, v)
        return attention_matrix


class DotProductAttention(dl.DeeplayModule):
    """Dot-product attention."""

    def __init__(self):
        """Initialize dot-product attention."""
        super().__init__()

    def forward(self, queries, keys, values):
        """Calculate dot-product attention."""
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (
            keys.size(-1) ** 0.5
        )
        attn_matrix = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_matrix, values)
        return attn_output, attn_matrix


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
    windowed_data = torch.tensor(data[0:144, :]).float()
    attention_matrix = GenerateAttentionMatrixes(14, 100)
    _, saved_am = attention_matrix(windowed_data)
    saved_am = saved_am.detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(saved_am)
    fig.colorbar(im)

    fig.savefig("../figures/problem2/attention_matrix_none.png")


if __name__ == "__main__":
    main()
