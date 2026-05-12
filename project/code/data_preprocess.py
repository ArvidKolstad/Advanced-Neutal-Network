import kagglehub as kh
import torch
import re
import string
import numpy as np
import pandas as pd
import fasttext.util
from torch.utils.data import Dataset


class SarcasmDataset(Dataset):
    def __init__(
        self,
        sentences: list,
        sarcasm_markers: np.ndarray,
        labels: np.ndarray,
    ):
        self.sentences = sentences
        self.sarcasm_markers = sarcasm_markers
        self.labels = labels
        self.embedder = load_fasttext()

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index):
        sentences_vector = self.embedder.get_sentence_vector(self.sentences[index])
        x = torch.tensor(
            np.concatenate((sentences_vector, self.sarcasm_markers[index]), axis=None)
        ).float()
        y = torch.tensor(self.labels[index]).unsqueeze(-1).float()
        return x, y


def load_fasttext():
    fasttext.util.download_model("en", if_exists="ignore")
    ft = fasttext.load_model("cc.en.300.bin")
    return ft


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


def extract_sarcasm_features(text_batch):
    batch_features = [
        {
            "all_caps_count": np.sum([1 for word in sample.split() if word.isupper()]),
            "exclamation_marks": sample.count("!"),
            "exclamation_question": sample.count("!?") + sample.count("?!"),
            "dot_dot_dot_counts": sample.count("..."),
        }
        for sample in text_batch
    ]
    return batch_features


def need_external_embedding(model):
    if model == "BERT":
        return False
    else:
        return True


def text_preprocess(data_frame: pd.DataFrame, model: str) -> pd.DataFrame:
    data_frame = data_frame.astype({"comment": str})
    data_frame.dropna(inplace=True)
    data_frame = data_frame.sample(frac=1, ignore_index=True)
    data_frame.reset_index(drop=True, inplace=True)

    if need_external_embedding(model):
        batch_features = extract_sarcasm_features(data_frame["comment"].to_list())

        data_frame["comment"] = data_frame["comment"].str.replace(
            f"[{re.escape(string.punctuation)}]", "", regex=True
        )
        data_frame["comment"] = data_frame["comment"].str.lower()

        batch_features = pd.DataFrame(batch_features)
        data_frame = pd.concat((data_frame, batch_features), axis=1)
    return data_frame


def main():
    wanted_columns = ["label", "comment"]
    df = load_dataset("reddit", wanted_columns, train=True)
    number_of_sarcastic_comments = len(df[df["label"] == 1])
    number_of_normal_comments = len(df[df["label"] == 0])

    # print(number_of_sarcastic_comments / number_of_normal_comments)

    # model = "log_reg"
    # text_preprocess(train_data, model)


if __name__ == "__main__":
    main()
