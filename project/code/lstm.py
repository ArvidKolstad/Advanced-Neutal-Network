import torch
import gc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
from typing import Optional
from train_utils import Logger, get_opt_threshold
from data_preprocess import (
    text_preprocess,
    create_dataset_fast_text,
    train_val_split,
    bilstm_collate,
    SarcasmDataset,
)
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from preformance_utils import get_f1_score


class LongShortTermMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_of_heads, marks_dim, threshold):
        super().__init__()

        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, number_of_heads, bias=False),
            nn.Softmax(dim=-2),
        )

        self.output_layer = nn.Linear(2 * hidden_dim * number_of_heads + marks_dim, 1)

    def forward(self, sentence, lengths, marks):

        packed = pack_padded_sequence(
            sentence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        H, _ = self.lstm(packed)
        H, _ = pad_packed_sequence(H, batch_first=True)
        Z = self.attention_layer(H)
        M = torch.matmul(Z.permute(0, 2, 1), H)
        concat = torch.cat((torch.flatten(M, start_dim=1), marks), dim=-1)

        output = self.output_layer(concat)
        return output

    def validate_model(self, val_loader: DataLoader, loss_function) -> tuple:
        self.eval()
        loss_val = 0.0
        f1_score = 0.0
        total_batches = len(val_loader)

        with torch.no_grad():
            for val_input, val_lengths, val_marks, val_target in val_loader:
                val_input, val_marks, val_target = (
                    val_input.to(self.device),
                    val_marks.to(self.device),
                    val_target.to(self.device).float(),
                )

                logits = self(val_input, val_lengths, val_marks).squeeze()
                loss = loss_function(logits, val_target)

                loss_val += loss.item()

                probs = torch.sigmoid(logits).detach().cpu()

                preds = (probs > self.threshold).int()

                f1_score += get_f1_score(preds, val_target.cpu())

        mean_val_loss = loss_val / total_batches
        mean_f1_score = f1_score / total_batches
        return mean_f1_score, mean_val_loss

    def train_epoch(self, train_loader, loss_function, optimizer):
        self.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        for train_input, train_lengths, train_marks, train_labels in train_loader:

            train_input, train_marks, train_labels = (
                train_input.to(self.device),
                train_marks.to(self.device),
                train_labels.to(self.device).float(),
            )
            optimizer.zero_grad(set_to_none=True)
            output = self(train_input, train_lengths, train_marks).squeeze()
            loss = loss_function(output, train_labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        mean_loss = total_loss / total_batches

        return mean_loss

    def train_params(
        self,
        max_epochs,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        scheduler,
        logger=True,
        stopper=12,
    ):
        writer: Optional[SummaryWriter] = None
        if logger:
            writer = Logger("BILSTM")

        min_val_loss = float("inf")
        epoch_number = 0
        epochs_with_increased_loss = 0

        print(f"training running on {self.device}")
        self.to(self.device)

        try:
            for epoch in range(max_epochs):
                print(f"Epoch: {epoch+1}")

                avg_loss = self.train_epoch(train_loader, loss_function, optimizer)
                avg_f1_score, avg_val_loss = self.validate_model(
                    val_loader, loss_function
                )

                if writer is not None:
                    writer.update_loss(avg_loss, avg_val_loss, epoch_number)
                    writer.update_f1_score(avg_f1_score, epoch_number)
                else:
                    print(
                        f"Training loss: {avg_loss:.4f}, Validation loss: {avg_val_loss:.4f}, F1 Score: {avg_f1_score:.4}"
                    )
                scheduler.step(avg_val_loss)

                if min_val_loss > avg_val_loss:
                    epochs_with_increased_loss = 0
                    min_val_loss = avg_val_loss
                    if writer is not None:
                        torch.save(
                            self.state_dict(),
                            os.path.join(writer.log_dir, "best_model.pth"),
                        )
                else:
                    epochs_with_increased_loss += 1

                if epochs_with_increased_loss == stopper:
                    print("training stopped early")
                    break
                epoch_number += 1

        finally:
            if writer is not None:
                writer.close()
        return min_val_loss


def kCV(
    model,
    k: int,
    training_data: list,
    model_params: dict,
    train_params: dict,
    optimizer,
    optimizer_settings: dict,
    scheduler,
    scheduler_settings,
):

    skf = StratifiedKFold(n_splits=k, shuffle=True)
    matrix, marks, labels = training_data
    loss_over_kcv = np.zeros(k)

    for idx, (train_index, val_index) in enumerate(skf.split(matrix, labels)):
        print(f"Now running fold {idx+1}/{k}")
        train_matrix, train_marks, train_labels = (
            matrix[train_index],
            marks[train_index],
            labels[train_index],
        )

        val_matrix, val_marks, val_labels = (
            matrix[val_index],
            marks[val_index],
            labels[val_index],
        )

        train_set = SarcasmDataset(train_matrix, train_marks, train_labels, "BILSTM")
        val_set = SarcasmDataset(val_matrix, val_marks, val_labels, "BILSTM")

        train_loader = DataLoader(
            train_set,
            batch_size=32,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            collate_fn=bilstm_collate,
            persistent_workers=False,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=64,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=bilstm_collate,
        )

        train_params["train_loader"] = train_loader

        train_params["val_loader"] = val_loader

        # 1. Create a local copy of train_params to avoid reference leaks
        local_train_params = train_params.copy()
        local_train_params["train_loader"] = train_loader
        local_train_params["val_loader"] = val_loader

        m = model(**model_params)
        optimizer_settings["params"] = m.parameters()

        # Create optimizer and scheduler locally
        opt = optimizer(**optimizer_settings)
        local_train_params["optimizer"] = opt

        sched_settings = scheduler_settings.copy()
        sched_settings["optimizer"] = opt
        local_train_params["scheduler"] = scheduler(**sched_settings)

        loss_over_kcv[idx] = m.train_params(**local_train_params)

        m.zero_grad(set_to_none=True)
        m.cpu()

        local_train_params.clear()

        del m, opt, train_loader, val_loader, local_train_params

        gc.collect()
        torch.cuda.empty_cache()
    mean_val_loss = np.mean(loss_over_kcv)
    std_val_loss = np.std(loss_over_kcv)

    return mean_val_loss, std_val_loss


def run_hyperparameter_opt():
    k = 5
    hyper_values = [1e-5, 1e-4, 1e-3, 1e-2]
    model_params = {
        "input_dim": 300,
        "hidden_dim": 128,
        "number_of_heads": 4,
        "marks_dim": 4,
        "threshold": 0.5,
    }
    train_params = {
        "max_epochs": 150,
        "train_loader": None,
        "val_loader": None,
        "loss_function": nn.BCEWithLogitsLoss(),
        "optimizer": None,
        "scheduler": None,
        "logger": False,
    }

    optimizer = torch.optim.Adam
    optimizer_settings = {"params": None, "lr": 0.001, "weight_decay": 0.001}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_settings = {
        "optimizer": None,
        "mode": "min",
        "factor": 0.2,
        "patience": 5,
    }
    df_train = pd.read_csv("./data/train.csv")
    df_train = text_preprocess(df_train, "BILSTM")
    comments = df_train["comment"].to_numpy()
    marks = df_train[
        [
            "all_caps_count",
            "exclamation_marks",
            "exclamation_question",
            "dot_dot_dot_counts",
        ]
    ].to_numpy()
    labels = df_train["label"].to_numpy()
    train_data = [comments, marks, labels]
    total_mean_val_loss = []
    total_std_val_loss = []

    for idx, value in enumerate(hyper_values):
        print(f"Now testing {idx+1}/{len(hyper_values)}")
        optimizer_settings["weight_decay"] = value
        mean_val_loss, std_val_loss = kCV(
            LongShortTermMemory,
            k,
            train_data,
            model_params,
            train_params,
            optimizer,
            optimizer_settings,
            scheduler,
            scheduler_settings,
        )
        total_mean_val_loss.append(mean_val_loss)
        total_std_val_loss.append(std_val_loss)

    np.savez(
        "./data/l2_hpv",
        x=hyper_values,
        mean=total_mean_val_loss,
        std=total_std_val_loss,
    )


def get_trained_model():
    max_epochs = 10
    model_params = {
        "input_dim": 300,
        "hidden_dim": 128,
        "number_of_heads": 4,
        "marks_dim": 4,
        "threshold": 0.5,
    }
    model = LongShortTermMemory(**model_params)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    val_frac = 0.33

    df_train = pd.read_csv("./data/train.csv")

    df_train = text_preprocess(df_train, "BILSTM")

    df_train, df_val = train_val_split(df_train, val_frac)

    data_set_train = create_dataset_fast_text(df_train, model="BILSTM")
    data_set_val = create_dataset_fast_text(df_val, model="BILSTM")

    train_loader = DataLoader(
        data_set_train,
        batch_size=32,
        num_workers=10,
        shuffle=True,
        pin_memory=True,
        collate_fn=bilstm_collate,
    )
    val_loader = DataLoader(
        data_set_val,
        batch_size=64,
        num_workers=10,
        shuffle=False,
        pin_memory=True,
        collate_fn=bilstm_collate,
    )

    train_settings = {
        "max_epochs": max_epochs,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "loss_function": loss_function,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "logger": True,
    }

    model.train_params(**train_settings)
    return model, val_loader


def plot_hyper_parameter(path):
    fig, ax = plt.subplots()

    data = np.load(path)
    hyper_values = [1e-5, 1e-4, 1e-3, 1e-2]
    # hyper_values = data["x"]
    mean, std = data["mean"], data["std"]

    ax.errorbar(hyper_values, mean, yerr=std)
    ax.set_xscale("log")
    ax.set_xlabel("L2 regularization")
    ax.set_ylabel("Validation loss")
    fig.savefig("../report/figures/hyper_l2reg.pdf")


def main():
    # run_hyperparameter_opt()
    plot_hyper_parameter("./data/l2_hpv.npz")
    model, val_loader = get_trained_model()

    model = model.to(torch.device("cuda"))

    best_thershold, best_f1_score = get_opt_threshold(
        model, val_loader, nn.BCEWithLogitsLoss()
    )
    print(f"f1 score {best_f1_score:.4}, threshold function {best_thershold}")

    """
    df_test = pd.read_csv("./data/test.csv")

    df_test = text_preprocess(df_test, "BILSTM")

    data_set_test = create_dataset_fast_text(df_test, model="BILSTM")
    test_loader = DataLoader(
        data_set_test,
        batch_size=64,
        num_workers=10,
        shuffle=False,
        pin_memory=True,
        collate_fn=bilstm_collate,
    )
    loss_function = nn.BCELoss()
    f1_score, loss_function = model.validate_model(test_loader, loss_function)
    print(f"f1 score {f1_score}, loss function {loss_function}")
    """


if __name__ == "__main__":
    main()
