from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
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


def get_opt_threshold(model, training_loader, loss_function):
    thresholds = np.arange(0, 1, 0.01)
    best_f1_score = 0.0
    best_threshold = 0.5
    for threshold in thresholds:
        print(f"Now testing for threshold: {threshold}")
        model.threshold = threshold
        f1_score, mean_val_score = model.validate_model(training_loader, loss_function)
        if best_f1_score < f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
    model.threshold = best_threshold

    return best_threshold, best_f1_score


def main():
    print("hello")


if __name__ == "__main__":
    main()
