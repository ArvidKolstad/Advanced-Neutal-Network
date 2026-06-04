import torch
from torchinfo import summary
from logistic_regression import LogisticRegression
from lstm import LongShortTermMemory
from bert import BERTweet


def main():
    model_params = {"dropout_rate": 0.1, "threshold": 0.5, "freeze_bert": False}

    model = BERTweet(**model_params)
    model.load_state_dict(
        torch.load("./runs/BERT_20260521_141613/best_model.pth", weights_only=True)
    )
    summary(model)


if __name__ == "__main__":
    main()
