import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from preformance_utils import get_f1_score, unpack_for_model
from datetime import datetime
import torch
import os


class Logger(SummaryWriter):
    def __init__(self, model_name):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join("runs", f"{model_name}_{self.timestamp}")
        super().__init__(log_dir=target_dir)
        self.log_dir = target_dir

        os.makedirs(self.log_dir, exist_ok=True)

    def update_loss(self, avg_loss, avg_val_loss, epoch_number):
        self.add_scalars(
            "Loss Functions training",
            {"Training": avg_loss, "Validation": avg_val_loss},
            epoch_number,
        )

    def update_f1_score(self, avg_f1_score, epoch_number):
        self.add_scalar("F1-score validation", avg_f1_score, epoch_number)


def train_val_split(
    df: pd.DataFrame, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_data_0_idx = df[df["label"] == 0].sample(frac=val_frac).index
    val_data_1_idx = df[df["label"] == 1].sample(frac=val_frac).index
    train_data = df.drop(index=val_data_0_idx)
    train_data = train_data.drop(index=val_data_1_idx)

    val_data = pd.concat(
        [df.loc[val_data_0_idx], df.loc[val_data_1_idx]], ignore_index=True
    ).sample(frac=1)
    return train_data, val_data


def get_opt_threshold(model, loader):
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            inputs, outputs = unpack_for_model(batch, model)
            inputs = [x.to(model.device) for x in inputs]

            logits = model(*inputs).squeeze()
            probs = torch.sigmoid(logits).detach().cpu()
            all_probs.append(probs)
            all_targets.append(outputs)

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    best_f1, best_threshold = 0.0, 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs > threshold).int()
        f1 = get_f1_score(preds, all_targets)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    model.threshold = best_threshold
    return best_threshold, best_f1


def main():
    print("hello")


if __name__ == "__main__":
    main()
