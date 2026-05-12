import torch.nn as nn
from typing import Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_preprocess import load_dataset, text_preprocess
from train_utils import Logger, train_val_split, create_dataset_fast_text
from torch.utils.tensorboard import SummaryWriter
from preformance_utils import get_f1_score
import os


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.layer = nn.Linear(input_dim, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        output = self.layer(x)
        return output

    def validate_model(self, val_loader: DataLoader, thresh_hold=0.5) -> float:
        self.eval()
        f1_score = 0.0
        total_batches = len(val_loader)

        with torch.no_grad():
            for val_input, val_target in val_loader:
                val_input, val_target = (
                    val_input.to(self.device),
                    val_target.int(),
                )

                logits = self(val_input)
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > thresh_hold).int()
                f1_score += get_f1_score(preds, val_target)

        mean_f1_score = f1_score / total_batches
        return mean_f1_score

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
            output = self(train_input)
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
    ):
        writer: Optional[SummaryWriter] = None
        if logger:
            writer = Logger()

        max_f1_score = 0
        epoch_number = 0

        print(f"training running on {self.device}")
        self.to(self.device)

        for epoch in range(max_epochs):
            print(f"Epoch: {epoch+1}")

            avg_loss = self.train_epoch(train_loader, loss_function, optimizer)
            avg_f1_score = self.validate_model(val_loader)

            if writer is not None:
                writer.update_loss(avg_loss, epoch_number)
                writer.update_f1_score(avg_f1_score, epoch_number)
            else:
                print(f"Training loss: {avg_loss}, Validation loss: {avg_f1_score}")
            scheduler.step(avg_f1_score)

            if (avg_f1_score < max_f1_score) and writer is not None:
                max_f1_score = avg_f1_score
                torch.save(
                    self.state_dict(), os.path.join(writer.log_dir, "best_model.pth")
                )
            epoch_number += 1

        if writer is not None:
            writer.close()


def main():
    lang_vec = 300
    max_epochs = 200

    sarcasm_markers_count = 4
    val_frac = 0.15

    input_dim = sarcasm_markers_count + lang_vec

    model = LogisticRegression(input_dim)
    df = load_dataset("reddit", ["comment", "label"])

    df = text_preprocess(df, "log_reg")
    df_train, df_val = train_val_split(df, val_frac)
    dataset_train = create_dataset_fast_text(df_train)
    dataset_val = create_dataset_fast_text(df_val)

    train_loader = DataLoader(
        dataset_train,
        batch_size=32,
        num_workers=10,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=64,
        num_workers=10,
        shuffle=True,
        pin_memory=True,
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
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


if __name__ == "__main__":
    main()
