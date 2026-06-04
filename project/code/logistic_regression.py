import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional
import torch
from torch.utils.data import DataLoader
from data_preprocess import text_preprocess, create_dataset_fast_text, train_val_split
from train_utils import Logger, get_opt_threshold
from torch.utils.tensorboard import SummaryWriter
from preformance_utils import get_f1_score, get_final_evaluation
import os


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, threshold=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.layer = nn.Linear(input_dim, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

    def __str__(self):
        return "logreg"

    def forward(self, x):
        output = self.layer(x)
        return output

    def validate_model(
        self, val_loader: DataLoader, loss_function
    ) -> tuple[float, float]:
        self.eval()
        f1_score = 0.0
        total_loss = 0.0
        total_batches = len(val_loader)

        with torch.no_grad():
            for val_input, val_target in val_loader:
                val_input, val_target = (
                    val_input.to(self.device),
                    val_target,
                )

                logits = self(val_input).cpu()
                loss = loss_function(logits, val_target)

                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > self.threshold).int()
                f1_score += get_f1_score(preds, val_target.int())

        mean_f1_score = f1_score / total_batches
        mean_val_loss = total_loss / total_batches

        return mean_f1_score, mean_val_loss

    def train_epoch(self, train_loader, loss_function, optimizer):
        self.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        for train_input, train_labels in train_loader:

            train_input, train_labels = (
                train_input.to(self.device),
                train_labels.to(self.device),
            )
            optimizer.zero_grad()
            logits = self(train_input)
            loss = loss_function(logits, train_labels)
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
    ):
        writer: Optional[SummaryWriter] = None
        if logger:
            writer = Logger("LogReg")

        max_loss = np.inf

        print(f"training running on {self.device}")
        self.to(self.device)

        for epoch in range(max_epochs):
            print(f"Epoch: {epoch+1}")

            avg_loss = self.train_epoch(train_loader, loss_function, optimizer)
            avg_f1_score, avg_val_loss = self.validate_model(val_loader, loss_function)

            if writer is not None:
                writer.update_loss(avg_loss, avg_val_loss, epoch)
                writer.update_f1_score(avg_f1_score, epoch)
            else:
                print(
                    f"Training loss: {avg_loss:.4f}, Validation loss: {avg_val_loss:.4f},F1-score: {avg_f1_score:.4f} "
                )
            scheduler.step(avg_val_loss)

            if (avg_val_loss < max_loss) and writer is not None:
                max_loss = avg_val_loss
                torch.save(
                    self.state_dict(), os.path.join(writer.log_dir, "best_model.pth")
                )

        if writer is not None:
            writer.close()


def main():
    lang_vec = 300
    max_epochs = 30

    sarcasm_markers_count = 4
    val_frac = 0.33

    input_dim = sarcasm_markers_count + lang_vec

    model = LogisticRegression(input_dim)
    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")

    df_train = text_preprocess(df_train, "log_reg")
    df_test = text_preprocess(df_test, "log_reg")

    df_train, df_val = train_val_split(df_train, val_frac)

    data_set_train = create_dataset_fast_text(df_train, model="log_reg")
    data_set_val = create_dataset_fast_text(df_val, model="log_reg")
    data_set_test = create_dataset_fast_text(df_test, model="log_reg")

    train_loader = DataLoader(
        data_set_train,
        batch_size=32,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        data_set_val,
        batch_size=64,
        num_workers=5,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
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

    del train_loader

    test_loader = DataLoader(
        data_set_test,
        batch_size=64,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    best_threshold, best_f1_score = get_opt_threshold(model, val_loader)

    print(
        f"Best threshold: {best_threshold:.3f}, with a F1-score of: {best_f1_score:.3f}"
    )
    model.threshold = best_threshold

    get_final_evaluation(model, test_loader)


if __name__ == "__main__":
    main()
