from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score


def get_confusion_matrix(pred, label, path, model_name):
    c_m = confusion_matrix(label, pred)
    cm_disp = ConfusionMatrixDisplay(
        c_m, display_labels=["Normal comment", "Sarcastic comment"]
    )
    cm_disp.figure_.savefig(path + "_" + model_name)


def get_f1_score(preds, labels):
    return f1_score(labels, preds)
