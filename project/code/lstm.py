import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
from typing import Optional
from train_utils import Logger, get_opt_threshold
from data_preprocess import (
    text_preprocess,
    create_dataset_fast_text,
    train_val_split,
    bilstm_collate,
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
        concated = torch.cat((torch.flatten(M, start_dim=1), marks), dim=-1)
        output = self.output_layer(concated)
        return output

    def validate_model(self, val_loader: DataLoader) -> float:
        self.eval()
        f1_score = 0.0
        total_batches = len(val_loader)

        with torch.no_grad():
            for val_input, val_lengths, val_marks, val_target in val_loader:
                val_input, val_marks, val_target = (
                    val_input.to(self.device),
                    val_marks.to(self.device),
                    val_target.int(),
                )

                logits = self(val_input, val_lengths, val_marks)
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > self.threshold).int()
                f1_score += get_f1_score(preds, val_target)

        mean_f1_score = f1_score / total_batches
        return mean_f1_score

    def train_epoch(self, train_loader, loss_function, optimizer):
        self.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        for train_input, train_lengths, train_marks, train_labels in train_loader:

            train_input, train_marks, train_labels = (
                train_input.to(self.device),
                train_marks.to(self.device),
                train_labels.to(self.device),
            )
            optimizer.zero_grad()
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
    ):
        writer: Optional[SummaryWriter] = None
        if logger:
            writer = Logger("BILSTM")

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

            if (avg_f1_score > max_f1_score) and writer is not None:
                max_f1_score = avg_f1_score
                torch.save(
                    self.state_dict(), os.path.join(writer.log_dir, "best_model.pth")
                )
            epoch_number += 1

        if writer is not None:
            writer.close()


def main():
    max_epochs = 200
    model_params = {
        "input_dim": 300,
        "hidden_dim": 128,
        "number_of_heads": 4,
        "marks_dim": 4,
        "threshold": 0.5,
    }
    model = LongShortTermMemory(**model_params)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    val_frac = 0.33

    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")

    df_train = text_preprocess(df_train, "BILSTM")
    df_test = text_preprocess(df_test, "BILSTM")

    sentences = df_train["comment"].tolist()

    df_train, df_val = train_val_split(df_train, val_frac)

    data_set_train = create_dataset_fast_text(df_train, model="BILSTM")
    data_set_val = create_dataset_fast_text(df_val, model="BILSTM")
    data_set_test = create_dataset_fast_text(df_test, model="BILSTM")

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
        shuffle=True,
        pin_memory=True,
        collate_fn=bilstm_collate,
    )

    test_loader = DataLoader(
        data_set_test,
        batch_size=64,
        num_workers=10,
        shuffle=True,
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


if __name__ == "__main__":
    main()
