import torch
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from preformance_utils import get_f1_score
from train_utils import Logger, get_opt_threshold
from torch.utils.tensorboard import SummaryWriter
from preformance_utils import get_final_evaluation
from data_preprocess import (
    text_preprocess,
    create_dataset_fast_text,
    train_val_split,
    SarcasmDataset,
)

import os
from typing import Optional


class BERTweet(nn.Module):
    def __init__(self, dropout_rate, threshold=0.5, freeze_bert=False):
        super(BERTweet, self).__init__()
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        if freeze_bert:
            for params in self.bertweet.parameters():
                params.requires_grad = False

    def __str__(self):
        return "BERTweet"

    def forward(self, x, attention_mask):
        output = self.bertweet(input_ids=x, attention_mask=attention_mask)
        cls_hidden_state = output[0][:, 0, :]
        layer = self.dropout(cls_hidden_state)
        output = self.classifier(layer)
        return output

    def validate_model(self, val_loader, loss_function):
        self.eval()
        loss_val = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for val_input, val_mask, val_target in val_loader:
                val_input, val_mask, val_target = (
                    val_input.to(self.device),
                    val_mask.to(self.device),
                    val_target.to(self.device).float(),
                )
                logits = self(val_input, val_mask).squeeze()
                loss_val += loss_function(logits, val_target).item()

                probs = torch.sigmoid(logits).detach().cpu()
                all_preds.append((probs > self.threshold).int())
                all_targets.append(val_target.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        mean_val_loss = loss_val / len(val_loader)
        f1 = get_f1_score(all_preds, all_targets)
        return f1, mean_val_loss

    def train_epoch(self, train_loader, loss_function, optimizer, scheduler):
        self.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        for train_input, train_mask, train_labels in train_loader:
            train_input, train_mask, train_labels = (
                train_input.to(self.device),
                train_mask.to(self.device),
                train_labels.to(self.device).float(),
            )
            optimizer.zero_grad(set_to_none=True)
            output = self(train_input, train_mask).squeeze()
            loss = loss_function(output, train_labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

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
        stopper=3,
    ):
        writer: Optional[SummaryWriter] = None
        if logger:
            writer = Logger("BERT")

        min_val_loss = float("inf")
        epochs_with_increased_loss = 0

        print(f"training running on {self.device}")
        self.to(self.device)

        try:
            for epoch in range(max_epochs):
                print(f"Epoch: {epoch+1}")

                avg_loss = self.train_epoch(
                    train_loader, loss_function, optimizer, scheduler
                )
                avg_f1_score, avg_val_loss = self.validate_model(
                    val_loader, loss_function
                )

                if writer is not None:
                    writer.update_loss(avg_loss, avg_val_loss, epoch)
                    writer.update_f1_score(avg_f1_score, epoch)
                else:
                    print(
                        f"Training loss: {avg_loss:.4f}, Validation loss: {avg_val_loss:.4f}, F1 Score: {avg_f1_score:.4}"
                    )

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
    matrix, labels = training_data
    loss_over_kcv = np.zeros(k)

    for idx, (train_index, val_index) in enumerate(skf.split(matrix, labels)):
        print(f"Now running fold {idx+1}/{k}")
        train_matrix, train_labels = (
            matrix[train_index],
            labels[train_index],
        )

        val_matrix, val_labels = (
            matrix[val_index],
            labels[val_index],
        )

        train_set = SarcasmDataset(train_matrix, None, train_labels, "BERT")
        val_set = SarcasmDataset(val_matrix, None, val_labels, "BERT")

        train_loader = DataLoader(
            train_set,
            batch_size=32,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            persistent_workers=False,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=64,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False,
        )

        train_params["train_loader"] = train_loader

        train_params["val_loader"] = val_loader

        local_train_params = train_params.copy()
        local_train_params["train_loader"] = train_loader
        local_train_params["val_loader"] = val_loader

        m = model(**model_params)
        optimizer_settings["params"] = m.parameters()

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

    optimizer = torch.optim.AdamW
    optimizer_settings = {"params": None, "lr": 0.001, "weight_decay": 0.001}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_settings = {
        "optimizer": None,
        "mode": "min",
        "factor": 0.5,
        "patience": 2,
    }
    df_train = pd.read_csv("./data/train.csv")
    df_train = text_preprocess(df_train, "BERT")
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
            BERTweet,
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


def get_llrd_param_groups(model, base_lr=5e-5, decay=0.8):

    param_groups = []
    param_groups.append(
        {"params": model.embeddings.parameters(), "lr": base_lr * (decay**12)}
    )

    for i, layer in enumerate(model.encoder.layer):
        layer_lr = base_lr * (decay ** (11 - i))
        param_groups.append({"params": layer.parameters(), "lr": layer_lr})
    param_groups.append({"params": model.pooler.parameters(), "lr": base_lr})

    return param_groups


def main():
    max_epochs = 5
    model_params = {"dropout_rate": 0.1, "threshold": 0.5, "freeze_bert": False}
    model = BERTweet(**model_params)
    param_groups = get_llrd_param_groups(model.bertweet)
    linear_layer = {"params": model.classifier.parameters(), "lr": 1e-3}
    param_groups.append(linear_layer)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(param_groups)

    val_frac = 0.33

    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")

    df_train = text_preprocess(df_train, "BERT")
    df_test = text_preprocess(df_test, "BERT")

    df_train, df_val = train_val_split(df_train, val_frac)

    data_set_train = create_dataset_fast_text(df_train, model="BERT")
    data_set_val = create_dataset_fast_text(df_val, model="BERT")
    data_set_test = create_dataset_fast_text(df_test, model="BERT")

    train_loader = DataLoader(
        data_set_train,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        data_set_val,
        batch_size=64,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        data_set_test,
        batch_size=64,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    num_training_steps_per_epoch = len(train_loader)
    total_training_steps = num_training_steps_per_epoch * max_epochs
    num_warmup_steps = int(0.10 * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
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

    get_opt_threshold(model, val_loader)
    get_final_evaluation(model, test_loader)


if __name__ == "__main__":
    main()
