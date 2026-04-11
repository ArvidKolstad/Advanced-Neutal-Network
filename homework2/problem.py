import numpy as np
import torchmetrics as tm
from torchmetrics.classification import BinaryF1Score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ImageDictDataset(Dataset):
    def __init__(self, data_dict):
        self.images = torch.tensor(np.array(data_dict["images"])).float()
        self.labels = torch.tensor(np.array(data_dict["labels"])).unsqueeze(2).float()
        print(self.images.shape)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].transpose(1, 0)
        particle = self.labels[idx, 0]
        position = self.labels[idx, 1:]

        return {"image": image, "particle": particle, "position": position}


class TrainingParticleCount(Dataset):

    def __init__(self, data_set):
        self.data = data_set

    def __len__(self):
        return len(self.data.images)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        image = torch.flatten(data_point["image"])
        particle = data_point["particle"]

        return image, particle


class MultilayerPerception(nn.Module):
    def __init__(self, layer_dim, act_func):
        super().__init__()
        self.layer_count = len(layer_dim)
        self.act_func_length = len(act_func)
        assert self.layer_count == self.act_func_length + 1

        layers = []
        for i in range(self.layer_count - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            layers.append(get_activation_function(act_func[i]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_activation_function(activation_name):
    if activation_name == "ReLU":
        return nn.ReLU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Activation name is wrong or not implemented")


def run_training(
    model,
    epochs,
    loss_function,
    optimizer,
    training_loader,
    validation_loader,
    file_name,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    best_validation_loss = np.inf

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        for inputs, labels in training_loader:
            inputs, labels = inputs.to(torch.device("cuda")), labels.to(
                torch.device("cuda")
            )
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_t_loss = running_loss / len(training_loader)
        avg_v_loss = validate_model(model, validation_loader, loss_function, device)
        if avg_v_loss < best_validation_loss:
            print(avg_v_loss)
            best_validation_loss = avg_v_loss
            torch.save(model.state_dict(), "./saved_models/" + file_name)
        print(
            f"Epoch {epoch +1}/{epochs}, Training loss: {avg_t_loss}, Validation loss: {avg_v_loss}"
        )


def validate_model(model, val_loader, loss_function, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
    avg_loss = val_loss / len(val_loader)
    return avg_loss


def plot_images(data_set, save_file):
    fig, axs = plt.subplots(3, 6, figsize=(16, 8))
    v_min, v_max = [0.0, 1.0]
    for ax in axs.ravel():
        data_point = data_set[np.random.randint(0, len(data_set))]
        image = data_point["image"]
        particle = data_point["particle"]
        position = data_point["position"]

        ax.imshow(image, vmin=v_min, vmax=v_max)
        if particle != 0:
            ax.set_title("Particle Exists")
            ax.scatter(position[0], position[1], label="Particle", c="red", s=10)
            ax.legend()
        else:
            ax.set_title("No Particle")
    fig.tight_layout()
    fig.savefig(save_file)


def plot_roc(model, loader, F1_classifier, file_name):
    """Plot ROC curve."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roc = tm.ROC(task="binary").to(device)
    F1_classifier = F1_classifier.to(device)

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.long().to(device)
        outputs = model(inputs)
        roc.update(outputs, labels)
        F1_classifier.update(outputs, labels)
    final_F1 = F1_classifier.compute()

    fig, ax = roc.plot(score=True)
    ax.grid(False)
    ax.axis("square")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="center right")
    ax.text(0.2, 0.2, f"F1 Score: {final_F1:.2f}")
    fig.tight_layout()
    fig.savefig(file_name)


def problem1():
    resolution = 64
    layers = [resolution * resolution, 3, 3, 1]

    act_funcs = ["ReLU", "ReLU", "sigmoid"]
    epochs = 15
    learning_rate = 0.001

    loss_function = nn.BCELoss()

    file_name = "mlp_easy_data_set"

    data = ImageDictDataset(pd.read_pickle("./simple_particle_dataset.pkl"))
    plot_images(data, "./figures/problem1/plot_training_set.png")

    data_particle_count = TrainingParticleCount(data)
    train, test = torch.utils.data.random_split(data_particle_count, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True, num_workers=15
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=15
    )

    mlp = MultilayerPerception(layers, act_funcs)
    print(mlp)

    optimizer = torch.optim.RMSprop(mlp.parameters(), lr=learning_rate)
    run_training(
        mlp, epochs, loss_function, optimizer, train_loader, test_loader, file_name
    )

    trained_model = MultilayerPerception(layers, act_funcs)
    trained_model.load_state_dict(torch.load("./saved_models/mlp_easy_data_set"))
    trained_model.to(torch.device("cuda"))

    F1_score = BinaryF1Score()
    plot_roc(trained_model, test_loader, F1_score, "./figures/problem1/roc_curve.png")


def main():
    problem1()


if __name__ == "__main__":

    main()
