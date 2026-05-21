import kagglehub as kh
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import re
import string
import numpy as np
import pandas as pd
import fasttext.util
from torch.utils.data import Dataset


def load_fasttext():
    fasttext.util.download_model("en", if_exists="ignore")
    ft = fasttext.load_model("cc.en.300.bin")
    return ft


run_fast_text = False


if run_fast_text:
    _FAST_TEXT = load_fasttext()
else:
    _FAST_TEXT = None


_BERT = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)


class SarcasmDataset(Dataset):
    def __init__(
        self,
        sentences: list[str],
        sarcasm_markers,
        labels: np.ndarray,
        model,
    ):
        self.sentences = sentences
        self.sarcasm_markers = sarcasm_markers
        self.labels = labels
        self.embedder_fasttext = _FAST_TEXT
        self.embedder_bert = _BERT
        self.model = model

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index):
        if self.model == "log_reg":
            sentences_vector = self.embedder_fasttext.get_sentence_vector(
                self.sentences[index]
            )
            x = torch.tensor(
                np.concatenate(
                    (sentences_vector, self.sarcasm_markers[index]), axis=None
                )
            ).float()
            y = torch.tensor(self.labels[index]).unsqueeze(-1).float()
            return x, y
        elif self.model == "BILSTM":
            words = self.sentences[index].split()
            scentences_matrix = torch.from_numpy(
                np.array([self.embedder_fasttext.get_word_vector(w) for w in words])
            ).float()
            markers = torch.tensor(self.sarcasm_markers[index]).float()
            labels = torch.tensor(self.labels[index]).float()
            return scentences_matrix, markers, labels
        elif self.model == "BERT":
            token = self.embedder_bert(
                self.sentences[index],
                padding="max_length",
                max_length=100,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = token["input_ids"].squeeze()
            attention_mask = token["attention_mask"].squeeze()

            label = self.labels[index]
            return input_ids, attention_mask, label


def bilstm_collate(
    batch,
):
    matrices, markers, labels = zip(*batch)
    padded_matrices = pad_sequence(list(matrices), batch_first=True, padding_value=0.0)
    lengths = torch.from_numpy(np.array([m.shape[0] for m in matrices]))
    markers = torch.stack(markers)
    labels = torch.stack(labels)
    return (padded_matrices, lengths, markers, labels)


def load_dataset(
    data_set_name: str,
    wanted_columns: list,
) -> pd.DataFrame:
    if data_set_name == "reddit":
        url = "danofer/sarcasm"
        file_name = "train-balanced-sarcasm.csv"
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


def create_dataset_fast_text(df: pd.DataFrame, model: str) -> Dataset:
    comments = df["comment"].to_list()
    labels = df["label"].to_numpy()
    if model != "BERT":
        marks = df[
            [
                "all_caps_count",
                "exclamation_marks",
                "exclamation_question",
                "dot_dot_dot_counts",
            ]
        ].to_numpy()
        dataset = SarcasmDataset(comments, marks, labels, model=model)
    else:
        dataset = SarcasmDataset(comments, None, labels, model=model)

    return dataset


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
    remove_idx = data_frame[
        data_frame["comment"].str.split().str.len() == 0
    ].index.tolist()
    data_frame = data_frame.drop(index=remove_idx)
    return data_frame


def main():
    print("Running data_preprocess.py")


if __name__ == "__main__":
    main()
