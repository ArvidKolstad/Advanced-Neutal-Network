from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


def get_confusion_matrix(pred, label, path):
    c_m = confusion_matrix(label, pred)
    cm_disp = ConfusionMatrixDisplay(
        c_m, display_labels=["Normal comment", "Sarcastic comment"]
    )
    cm_disp.figure_.savefig(path + "_confusion_matrix.pdf")


def get_f1_score(preds, labels):
    return f1_score(labels, preds)


def unpack_for_model(batch, model) -> tuple[list[torch.Tensor], torch.Tensor]:
    from lstm import LongShortTermMemory
    from bert import BERTweet
    from logistic_regression import LogisticRegression

    if type(model) == LogisticRegression:
        inputs, target = batch
        inputs = [inputs]
    elif type(model) == LongShortTermMemory:
        sentence_matrix, markers, target = batch
        inputs = [sentence_matrix, markers]
    elif type(model) == BERTweet:
        input_ids, attention_mask, target = batch
        inputs = [input_ids, attention_mask]
    else:
        raise ValueError("model is not implemented")
    return inputs, target


def get_final_evaluation(model, test_loader):
    model.eval()
    preds = []
    targets = []
    for batch in test_loader:
        inputs, target = unpack_for_model(batch, model)
        targets.append(target)
        inputs = [x.to(model.device) for x in inputs]
        preds.append(model(*inputs).squeeze().detach().numpy())
    targets = np.array(targets).flatten()
    preds = np.array(preds).flatten()

    accuracy = np.mean(targets == preds)
    f1_score = get_f1_score(preds, targets)
    path = f"../report/figures/{model}"
    get_confusion_matrix(preds, targets, path)
    scores = {"Accuracy": accuracy, "F1-Score": f1_score, "Threshold": model.threshold}

    model_data = pd.DataFrame(scores).reset_index(drop=True)
    fig, ax = plt.subplots()
    ax.table(model_data, cellLoc="center")
    fig.tight_layout()
    fig.savefig(path + "_data")
