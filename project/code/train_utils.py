from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt


def kCV(
    k,
    model_class,
    train_input,
    train_target,
    model_settings,
    training_settings,
    dataset: Dataset,
    save_model=False,
    model_name="normal",
    sarcasm_markers=None,
):

    skf = StratifiedKFold(n_splits=k, shuffle=True)
    best_fold_score = 0.0
    fold_score = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_input, train_target)):
        print(f"Fold: {fold +1 }/{k}")
        train_batch = {
            "scentences": train_input[train_idx],
            "labels": train_target[train_idx],
        }
        val_batch = {
            "scentences": train_input[val_idx],
            "labels": train_target[val_idx],
        }

        if sarcasm_markers:
            train_batch["sarcasm_markers"] = sarcasm_markers[val_idx]
            val_batch["sarcasm_markers"] = sarcasm_markers[val_idx]

        train_data_set = dataset(**train_batch)
        val_data_set = dataset(**val_batch)

        train_batch = DataLoader(
            train_data_set, batch_size=32, num_workers=10, shuffle=True, pin_memory=True
        )

        val_batch = DataLoader(
            val_data_set, batch_size=64, num_workers=10, shuffle=True, pin_memory=True
        )

        training_settings["train_data"] = train_batch
        training_settings["val_data"] = val_batch

        model = model_class(**model_settings)

        score = model.train_params(**training_settings)

        if score > best_fold_score and save_model:
            best_fold_score = score

            model.save("./saved_models/" + model_name)
            model.save_model_settings(model_name)

        fold_score.append(score)
        print(f"Best val score: {score}")
    return fold_score


def get_dict(module):
    if module == "model":
        return 0
    elif module == "train":
        return 1
    elif module == "kCV":
        return 2
    else:
        raise ValueError("Module doesn't exist")


def hyper_parameter_opt(
    hyper_parameter: str,
    parameter_values: list,
    module: str,
    params: list,
    model_name: str,
    data_matrix=None,
    data_label=None,
):

    print("Now training for " + hyper_parameter)
    if isinstance(params[get_dict(module)][hyper_parameter], (list, np.ndarray)):
        original_value = params[get_dict(module)][hyper_parameter].copy()
    else:
        original_value = params[get_dict(module)][hyper_parameter]

    original_data_set = params[get_dict("data")]
    color = ["red", "blue"]
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    if data_matrix is None and data_label is None:
        redo_data = False
    else:
        redo_data = True

    input_dim = None
    x1 = None
    value = None
    scores = []
    x2 = []

    for idx, value in enumerate(parameter_values):

        params[get_dict(module)][hyper_parameter] = value
        score = kCV(**params[get_dict("kCV")])
        scores.append(score)
        if hyper_parameter == "layer_dim":
            x1 = np.ones_like(score) * input_dim
            x2.append(input_dim)
        else:
            x1 = np.ones_like(score) * value
            x2.append(value)

        ax1.scatter(x1, score, color=color[idx % 2])

        print(f"Done hyper training: {idx+1}/{len(parameter_values)}")

    ax2.boxplot(scores, tick_labels=x2)
    ax2.set_title(f"Hyper parameter: {hyper_parameter}")
    fig2.savefig(
        "../figures/hyper_param_opt/" + hyper_parameter + "_" + model_name + "_boxplot"
    )

    ax1.grid()
    ax1.set_title(f"Hyper parameter: {hyper_parameter}")
    fig1.tight_layout()
    fig1.savefig("../figures/hyper_param_opt/" + hyper_parameter + "_" + model_name)
    params[get_dict(module)][hyper_parameter] = original_value

    if redo_data:
        params[get_dict("data")] = original_data_set


def main():
    print("hello")


if __name__ == "__main__":
    main()
