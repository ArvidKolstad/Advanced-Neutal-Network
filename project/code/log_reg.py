import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


class LogRegTfIdf(LogisticRegression):
    def __init__(self, model_settings: dict):
        self.settings = model_settings
        self.model = LogisticRegression(**model_settings)

    def clone(self) -> LogisticRegression:
        cloned_model = LogRegTfIdf(**self.settings)
        return cloned_model


def cv_train(model, folds, train_data):
    input_data, label_data = train_data
    kf = KFold(n_splits=folds, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(input_data, y=label_data)):
        running_model = model.clone()
        print(f"Now running fold: {fold}")
        running_model.fit(input_data[train_idx], label_data[train_idx])
        running_model.predict(input_data[test_idx], label_data[test_idx])


def main():
    print("Hello")


if __name__ == "__main__":
    main()
