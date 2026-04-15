import numpy as np
import torchmetrics as tm
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class ImageDictDataset(Dataset):
    def __init__(self, data_dict):
        self.images = torch.tensor(np.array(data_dict["images"])).float()
        self.labels = torch.tensor(np.array(data_dict["labels"])).unsqueeze(2).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].transpose(1, 0)
        particle = self.labels[idx, 0]
        position = self.labels[idx, 1:]

        return {"image": image, "particle": particle, "position": position}


class TrainingParticleCount(Dataset):

    def __init__(self, data_set, flat=True, transform=None):
        self.data = data_set
        self.flat = flat
        self.transform = transform

    def __len__(self):
        return len(self.data.images)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        if self.flat:
            image = torch.flatten(data_point["image"])
        else:
            image = data_point["image"]
            image = torch.unsqueeze(image, 0)

        if self.transform:
            image = self.transform(image)

        particle = data_point["particle"]

        return image, particle


class TrainingParticlePosition(Dataset):

    def __init__(self, data_set, flat=True, transform=None):
        self.data = data_set
        self.flat = flat
        self.transform = transform

    def __len__(self):
        return len(self.data.images)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        if self.flat:
            image = torch.flatten(data_point["image"])
        else:
            image = data_point["image"]
            image = torch.unsqueeze(image, 0)

        if self.transform:
            image = self.transform(image)

        particle = data_point["position"]

        return image, particle


def get_mean_and_std_input(loader):
    mean = 0.0
    std = 0.0
    number_of_batches = len(loader)
    for images, _ in loader:
        mean += torch.mean(images)
        std += torch.std(images)
    return mean / number_of_batches, std / number_of_batches


class MultilayerPerception(nn.Module):
    def __init__(self, layer_dim, act_func, dropout_rate=0.3):
        super().__init__()
        self.layer_count = len(layer_dim)
        self.act_func_length = len(act_func)
        assert self.layer_count == self.act_func_length + 1

        layers = []
        for i in range(self.layer_count - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            layers.append(get_activation_function(act_func[i]))
            if i < self.layer_count - 2:
                layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(
        self, convolution_layout, mlp_layout, activation_functions, dropout_rate=0.3
    ):
        super().__init__()
        conv_layers = []
        for idx in range(len(convolution_layout) - 1):
            conv_layers.append(
                nn.Conv2d(
                    convolution_layout[idx],
                    convolution_layout[idx + 1],
                    3,
                    padding="same",
                )
            )
            conv_layers.append(nn.BatchNorm2d(convolution_layout[idx + 1]))
            conv_layers.append(nn.ReLU())

            conv_layers.append(
                nn.Conv2d(
                    convolution_layout[idx + 1],
                    convolution_layout[idx + 1],
                    3,
                    padding="same",
                )
            )
            conv_layers.append(nn.BatchNorm2d(convolution_layout[idx + 1]))
            conv_layers.append(nn.ReLU())

            conv_layers.append(nn.MaxPool2d(2, 2))

        self.convolution_module = nn.Sequential(*conv_layers)

        self.connector = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = MultilayerPerception(
            mlp_layout, activation_functions, dropout_rate=dropout_rate
        )

    def forward(self, inputs):
        x = inputs
        x = self.convolution_module(x)
        x = self.connector(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_activation_function(activation_name):
    if activation_name == "ReLU":
        return nn.ReLU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "softmax":
        return nn.Softmax()
    elif activation_name == "logsoftmax":
        return nn.LogSoftmax()
    elif activation_name == "identity":
        return nn.Identity()
    else:
        raise ValueError("Activation name is wrong or not implemented")


def prepare_labels(labels, loss_function):
    if isinstance(loss_function, nn.BCELoss):
        return labels.squeeze(1).float()
    if isinstance(loss_function, nn.CrossEntropyLoss):
        return labels.squeeze(1).long()
    if isinstance(loss_function, nn.L1Loss):
        return labels.squeeze(2).float()


def run_training(
    model,
    epochs,
    loss_function,
    optimizer,
    training_loader,
    validation_loader,
    file_name,
    scheduler=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    best_validation_loss = np.inf

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        for inputs, labels in training_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = prepare_labels(labels, loss_function)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_t_loss = running_loss / len(training_loader)
        avg_v_loss = validate_model(model, validation_loader, loss_function, device)
        if scheduler:
            scheduler.step(avg_v_loss)
        if avg_v_loss < best_validation_loss:
            best_validation_loss = avg_v_loss
            torch.save(model.state_dict(), "./saved_models/" + file_name)
        print(
            f"Epoch {epoch +1}/{epochs}, Training loss: {avg_t_loss:.5f}, Validation loss: {avg_v_loss:.5f}"
        )
    return best_validation_loss


def validate_model(model, val_loader, loss_function, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = prepare_labels(labels, loss_function)
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(1), labels)
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


def plot_roc_binary(model, loader, loss_function, file_name):
    """Plot ROC curve."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roc = tm.ROC(task="binary").to(device)
    F1_classifier = BinaryF1Score().to(device)

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = prepare_labels(labels, loss_function)
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


def plot_roc_multiclass(model, loader, number_of_classes, file_name):
    """Plot ROC curve."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roc = tm.ROC(task="multiclass", num_classes=number_of_classes).to(device)
    F1_classifier = MulticlassF1Score(number_of_classes).to(device)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.long().to(device).squeeze()
        outputs = model(inputs)

        roc.update(torch.softmax(outputs, dim=1), labels)
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


def plot_predictions(model, test_batch, save_file, device, val_loss):
    indices = np.random.choice(np.arange(len(test_batch[0])), 6, replace=False)
    images = [test_batch[0][index].to(device) for index in indices]

    annotations = [test_batch[1][index] for index in indices]
    model.eval()
    inputs = torch.stack(images)

    predictions = model(inputs)

    fig, axs = plt.subplots(1, 6, figsize=(25, 8))
    for ax, im, ann, pred in zip(axs, images, annotations, predictions):
        im = im.cpu()
        pred = pred.cpu()
        im_to_show = im.permute(1, 2, 0).numpy().squeeze()
        ax.imshow(im_to_show, cmap="gray")

        ax.scatter(
            ann[0], ann[1], marker="+", c="g", s=500, linewidth=6, label="Annotation"
        )

        pred = pred.detach().numpy()

        ax.scatter(
            pred[0], pred[1], marker="x", c="r", s=500, linewidth=4, label="Prediction"
        )

        ax.set_axis_off()
    ax.legend(loc=(0.5, 0.8), framealpha=1, fontsize=24)
    fig.suptitle(f"Validation loss after training: {val_loss:.3f}")
    fig.tight_layout()
    fig.savefig(save_file)


def main():
    print("Running helper file")


if __name__ == "__main__":
    main()
