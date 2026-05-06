import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import seaborn as sns
from torch_geometric.nn.norm import LayerNorm
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import deeplay as dl
from torch.utils.data import Dataset, DataLoader


class WeatherDataset(Dataset):
    def __init__(self, data, input_len, horizon):
        self.input_len = input_len
        self.horizon = horizon
        self.data = torch.tensor(data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.input_len - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len + self.horizon - 1, 0]

        return x, y


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

    def forward(
        self,
        in_sequence,
    ):
        """Apply the multi-head attention mechanism to the input sequence."""
        batch_size, seq_len, embed_dim = in_sequence.shape
        Q = self.Wq(in_sequence)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        K = self.Wk(in_sequence)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        V = self.Wv(in_sequence)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=in_sequence.device)
        ).bool()

        attn_scores = attn_scores.masked_fill(mask == False, float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        if not self.training:
            self.attn_weights = attn_weights.cpu().detach().numpy()

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)
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

    def forward(self, in_sequence):
        """Refine sequence via attention and feedforward layers."""
        attns = self.self_attn(in_sequence)
        attns = self.attn_dropout(attns)
        attns = self.attn_skip(in_sequence, attns)
        attns = self.attn_norm(attns)

        out_sequence = self.feedforward(attns)
        out_sequence = self.feedforward_dropout(out_sequence)
        out_sequence = self.feedforward_skip(attns, out_sequence)
        out_sequence = self.feedforward_norm(out_sequence)

        return out_sequence


class TransformerEncoderModel(dl.DeeplayModule):
    """Transformer encoder model."""

    def __init__(
        self,
        input_features,
        num_features,
        num_heads,
        feedforward_dim,
        num_layers,
        out_dim,
        dropout=0.0,
    ):
        """Initialize transformer encoder model."""
        super().__init__()
        self.num_features = num_features

        self.embedding = dl.Layer(torch.nn.Linear, input_features, num_features)

        self.pos_encoder = periodic_encoding

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
        )

    def forward(self, inputs):
        """Predict sentiment of movie reviews."""

        embeddings = self.embedding(inputs)
        embedding_dim = embeddings.shape[-1]
        seq_len = inputs.shape[1]

        pos_embeddings = self.pos_encoder(seq_len, embedding_dim)

        out_sequence = pos_embeddings + embeddings
        for transformer_layer in self.transformer_block:
            out_sequence = transformer_layer(out_sequence)

        out = out_sequence[:, -1, :]
        pred = self.out_block(out).squeeze()

        return pred


def periodic_encoding(time_steps, embedding_dim):
    encoding = []
    device = torch.device("cuda")
    i = np.arange(0, time_steps, 1)
    for j in range(0, embedding_dim):
        if j % 2 == 0:
            encoding.append(np.sin(i / (10000 ** (j / embedding_dim))))
        else:
            encoding.append(np.cos(i / (10000 ** (j / embedding_dim))))
    encoding = torch.tensor(np.vstack(encoding)).permute(1, 0).float().to(device)

    return encoding


def plot_covariance_matrix(covar_matrix, file_path):

    fig, _ = plt.subplots(figsize=(10, 10))
    sns.heatmap(covar_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    fig.tight_layout()
    fig.savefig(file_path)


def define_benchmark(training_data):
    daily_samples = 144
    lag = 12
    temperature = training_data["T (degC)"].to_numpy()
    benchmark_celsius = np.mean(
        np.abs(
            temperature[daily_samples + lag :: daily_samples]
            - temperature[lag : -(daily_samples - lag) : daily_samples]
        )
    )
    std_temp = np.std(temperature)
    benchmark = benchmark_celsius / std_temp
    return benchmark


def preprocess_data(dataframe):

    daily_samples = 144
    future_prediction = 72
    normalized_dataframe = (dataframe - dataframe.mean()) / (dataframe.std())
    covariance_matrix = normalized_dataframe.cov()
    plot_covariance_matrix(
        covariance_matrix, "../figures/problem3/covariance_matrix_org.png"
    )

    dataframe.drop(
        labels=[
            "Tpot (K)",
            "Tdew (degC)",
            "VPact (mbar)",
            "sh (g/kg)",
            "max. wv (m/s)",
            "p (mbar)",
            "wv (m/s)",
            "wd (deg)",
        ],
        axis=1,
        inplace=True,
    )
    normalized_dataframe = (dataframe - dataframe.mean()) / (dataframe.std())

    plot_covariance_matrix(
        normalized_dataframe.cov(), "../figures/problem3/covar_matrix_dropped.png"
    )

    n = len(dataframe)
    training_end = int(0.90 * n)

    train_data = dataframe[:training_end]
    val_data = dataframe[training_end:]

    mean = train_data.mean()
    std = train_data.std()
    temperature_scalar = {"mean": mean, "std": std}

    norm_train_data = (train_data - mean) / std
    norm_val_data = (val_data - mean) / std

    bench_mark = define_benchmark(norm_val_data)

    train_set = WeatherDataset(norm_train_data, daily_samples, future_prediction)
    val_set = WeatherDataset(norm_val_data, daily_samples, future_prediction)

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        num_workers=10,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, bench_mark, temperature_scalar


def train_model():
    dataframe = pd.read_csv("jena_climate_2009_2016.csv", index_col=0)
    train_data, val_data, bench_mark, temp_scale = preprocess_data(dataframe)
    in_features = 6
    m = 40
    heads = 4
    feed_forward_dim = m
    layers = 2
    out_dim = 1
    drop_out = 0.2

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-weather-transformer",
        save_top_k=1,
        mode="min",
    )
    model = TransformerEncoderModel(
        in_features, m, heads, feed_forward_dim, layers, out_dim, dropout=drop_out
    ).create()

    model_application = dl.Regressor(
        model=model,
        loss=torch.nn.L1Loss(),
        optimizer=dl.AdamW(lr=1e-4),
    ).create()

    trainer = dl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        callbacks=[early_stop, checkpoint],
        logger=CSVLogger("checkpoints/"),
    )

    trainer.fit(model_application, train_data, val_data)


def plot_training(path, bench_mark, temp_scale):
    df = pd.read_csv(path)
    train_loss = df["train_loss_epoch"].dropna().to_numpy()
    val_loss = df["val_loss_epoch"].dropna().to_numpy()
    max_epoch = len(val_loss)

    epochs = np.arange(1, max_epoch + 1, 1)
    mean_temp = temp_scale["mean"]["T (degC)"]
    std_temp = temp_scale["std"]["T (degC)"]

    train_loss = std_temp * train_loss + mean_temp
    val_loss = std_temp * val_loss + mean_temp
    bench_mark = std_temp * bench_mark + mean_temp

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label="Training loss")
    ax.plot(epochs, val_loss, label="Validation loss")
    ax.axhline(bench_mark, label="benchmark", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE in temperature (Celsius)")
    fig.tight_layout()
    fig.savefig("../figures/problem3/trainig_loss.png")


def plot_attention_matrix(attention):
    fig, ax = plt.subplots(4, 4, figsize=(15, 15))

    for i in range(4):
        for j in range(4):
            im = ax[i, j].imshow(attention[i, j])
            ax[i, j].set_title(f"Sample {1}, Head {j}")
            fig.colorbar(im, ax=ax[i, j])
    fig.tight_layout()
    fig.savefig("../figures/problem3/attention_matrix.png")


def main():

    dataframe = pd.read_csv("jena_climate_2009_2016.csv", index_col=0)
    train_data, val_data, bench_mark, temp_scale = preprocess_data(dataframe)

    # train_model()
    model_application = dl.Regressor.load_from_checkpoint(
        "./checkpoints/best-weather-transformer.ckpt"
    )
    model_application.eval()
    with torch.no_grad():
        x, _ = next(iter(val_data))
        x = x.to(torch.device("cuda"))
        model_application(x)
    attention = model_application.model.transformer_block[0].self_attn.attn_weights
    plot_attention_matrix(attention)

    plot_training(
        "./checkpoints/lightning_logs/version_0/metrics.csv", bench_mark, temp_scale
    )


if __name__ == "__main__":
    main()
