import kagglehub as kh
import numpy as np
import pandas as pd


def load_dataset(
    data_set_name: str,
    wanted_columns: list,
    train: bool = True,
) -> pd.DataFrame:
    if data_set_name == "reddit":
        url = "danofer/sarcasm"
        if train:
            file_name = "train-balanced-sarcasm.csv"
        else:
            file_name = "test_balanced.csv"
    else:
        raise ValueError("No data set with that name is implemented")
    df = kh.dataset_load(
        kh.KaggleDatasetAdapter.PANDAS,
        url,
        file_name,
    )
    df = df[wanted_columns]

    return df


def main():
    wanted_columns = ["label", "comment"]
    train_data = load_dataset(
        "reddit",
        wanted_columns,
        train=True,
    )
    print(train_data.head(10))


if __name__ == "__main__":
    main()
