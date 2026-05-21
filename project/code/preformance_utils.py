from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


def get_confusion_matrix(pred, label, path):
    c_m = confusion_matrix(label, pred, normalize="all")
    cm_disp = ConfusionMatrixDisplay(c_m, display_labels=["Normal", "Sarcastic"])
    cm_disp.plot()
    cm_disp.figure_.savefig(path + "_confusion_matrix.pdf")


def get_f1_score(preds, labels):
    return f1_score(labels, preds)


def unpack_for_model(batch, model) -> tuple[list[torch.Tensor], torch.Tensor]:
    model_name = type(model).__name__

    if model_name == "LogisticRegression":
        inputs, target = batch
        inputs = [inputs]
    elif model_name == "LongShortTermMemory":
        sentence_matrix, lengths, markers, target = batch
        inputs = [sentence_matrix, lengths, markers]
    elif model_name == "BERTweet":
        input_ids, attention_mask, target = batch
        inputs = [input_ids, attention_mask]
    else:
        raise ValueError("model is not implemented")
    return inputs, target


def get_final_evaluation(model, test_loader):
    model.to(model.device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, target = unpack_for_model(batch, model)

            targets.append(target.cpu().numpy().reshape(-1))

            inputs = [x.to(model.device) for x in inputs]

            logits = model(*inputs).reshape(-1)

            probabilities = torch.sigmoid(logits)

            binary_preds = (
                (probabilities >= model.threshold).cpu().numpy().astype(np.int8)
            )
            preds.append(binary_preds)

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)

    accuracy = np.mean(targets == preds)
    f1_score = get_f1_score(preds, targets)

    path = f"../report/figures/{model}"
    get_confusion_matrix(preds, targets, path)

    scores = {
        "Accuracy": [accuracy],
        "F1-Score": [f1_score],
        "Threshold": [model.threshold],
    }

    model_data = pd.DataFrame(scores)
    fig, ax = plt.subplots()
    ax.table(
        cellText=model_data.values,
        colLabels=model_data.columns,
        cellLoc="center",
        loc="center",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path + "_data.png")
    plt.close(fig)

    return scores
