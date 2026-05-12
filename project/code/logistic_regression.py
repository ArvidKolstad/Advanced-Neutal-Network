import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_preprocess import SarcasmDataset, load_dataset, text_preprocess


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layer = nn.Linear(input_dim, 1)
        self.act_func = nn.Sigmoid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        hidden = self.layer(x)
        output = self.act_func(hidden)
        return output

    def validate_model(self, val_loader: DataLoader, loss_function) -> float:
        self.eval()
        loss = 0.0
        total_batches = len(val_loader)

        for val_input, val_target in val_loader:
            output = self(val_input)
            loss += loss_function(output, val_target)

        mean_loss = loss / total_batches
        return mean_loss

    def train_epoch(self, train_loader, loss_function, optimizer):
        self.train()

        for train_input, train_labels in train_loader:

            train_input, train_labels = train_input.to(self.device), train_labels.to(
                self.device
            )
            optimizer.zero_grad()
            output = self(train_input)
            loss = loss_function(output, train_labels)
            loss.backwards()
            optimizer.step()

    def train_params(
        self, max_epochs, train_loader, val_loader, loss_function, optimizer, scheduler
    ):
        print(f"training running on {self.device}")
        self.to(self.device)
        for epoch in range(max_epochs):
            print(epoch)
            self.train_epoch(train_loader, loss_function, optimizer)
        val_score = self.validate_model(val_loader, loss_function)
        scheduler.step(val_score)


def main():
    lang_vec = 300
    sarcasm_markers_count = 4

    input_dim = sarcasm_markers_count + lang_vec

    model = LogisticRegression(input_dim)
    df = load_dataset("reddit", ["comment", "label"])
    df = text_preprocess(df, "log_reg")
    comments = df["comment"].to_list()
    labels = df["label"].to_numpy()
    marks = df[
        "all_caps_count",
        "exclamation_marks",
        "exclamation_question",
        "dot_dot_dot_counts",
    ].to_numpy()

    dataset = SarcasmDataset(comments, labels, marks)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=10,
        shuffle=True,
        pin_memory=True,
    )

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )


if __name__ == "__main__":
    main()
