import lazy_import

_orig = lazy_import.LazyModule.__getattribute__


def _patched(self, name):
    if name == "__file__":
        return None
    return _orig(self, name)


lazy_import.LazyModule.__getattribute__ = _patched

import torch
import deeplay as dl
import deeptrack as dt
import numpy as np
from kornia.geometry.transform import translate, rotate, get_rotation_matrix2d
import matplotlib.pyplot as plt
from numpy.random import uniform
from torch import rand


class ParticleLocalizer(dl.Application):
    """LodeSTAR implementation with translations."""

    def __init__(self, model, image_size, n_transforms=8, **kwargs):
        """Initialize the ParticleLocalizer."""
        self.model, self.n_transforms = model, n_transforms
        self.image_size = image_size
        super().__init__(**kwargs)

    def forward(self, batch):
        """Forward pass through the model."""
        return self.model(batch)

    def random_arguments(self):
        """Generate random arguments for transformations."""
        return {
            "translation": (
                rand(self.n_transforms, 2).float().to(self.device) * 5 - 2.5
            ),
            "rotation": rand(self.n_transforms).float().to(self.device) * 360 - 180,
            "image_size": self.image_size,
        }

    def forward_transform(self, batch, translation, rotation, image_size):
        translated_image = image_translation(batch, translation)
        transformed_image = image_rotation(translated_image, rotation)
        return transformed_image

    def inverse_transform(self, preds, translation, rotation, image_size):
        translated_pred = inverse_rotation(preds, rotation, image_size)
        default_pred = inverse_translation(translated_pred, translation)
        return default_pred

    def training_step(self, image, batch_idx):
        """Perform a single training step."""
        image, *_ = image
        batch = image.repeat(self.n_transforms, 1, 1, 1)

        kwargs = self.random_arguments()
        transformed_batch = self.forward_transform(batch, **kwargs)

        pred_position = self(transformed_batch)
        pred_position = self.inverse_transform(pred_position, **kwargs)

        average_pred_position = pred_position.mean(dim=0, keepdim=True).repeat(
            self.n_transforms, 1
        )
        loss = self.loss(pred_position, average_pred_position)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


class ParticleLocalizerWithFlips(ParticleLocalizer):
    """ParticleLocalizer with additional flips."""

    def forward_transform(self, batch, translation, flip_x, flip_y):
        """Apply forward translations and flips to the batch."""
        batch = image_translation(batch, translation)
        batch = flip_transform(batch, flip_x, dim=3)
        batch = flip_transform(batch, flip_y, dim=2)
        return batch

    def inverse_transform(self, preds, translation, flip_x, flip_y):
        """Apply the inverse transformation to the predictions."""
        preds = inverse_flip_transform(preds, flip_x, dim=1)
        preds = inverse_flip_transform(preds, flip_y, dim=0)
        preds = inverse_translation(preds, translation)
        return preds

    def random_arguments(self):
        """Generate random arguments for translations and flips."""
        return {
            "translation": (
                rand(self.n_transforms, 2).float().to(self.device) * 5 - 2.5
            ),
            "flip_x": rand(self.n_transforms).float().to(self.device) > 0.5,
            "flip_y": rand(self.n_transforms).float().to(self.device) > 0.5,
        }


def flip_transform(batch, should_flip, dim):
    """Conditionally flip batch along a specified dimension."""
    should_flip = should_flip.view(-1, 1, 1, 1)
    return torch.where(should_flip, batch.flip(dims=(dim,)), batch)


def inverse_flip_transform(preds, should_flip, dim):
    """Conditionally inverse flip transformation based on should flip."""
    should_flip_mask = torch.zeros_like(preds).bool()
    should_flip_mask[should_flip, dim] = 1
    return torch.where(should_flip_mask, -preds, preds)


def image_translation(batch, translation):
    """Translate a batch of images."""
    xy_flipped_translation = translation[:, [1, 0]]
    return translate(batch, xy_flipped_translation, padding_mode="reflection")


def image_rotation(batch, rotation):
    return rotate(batch, rotation, padding_mode="reflection")


def inverse_translation(preds, applied_translation):
    """Invert translation of predicted positions."""
    return preds - applied_translation


def inverse_rotation(preds, applied_rotation, image_size):
    rad = applied_rotation * (np.pi / 180.0)
    center = (image_size - 1) / 2

    p = preds - center

    new_x = p[:, 1] * torch.cos(rad) - p[:, 0] * torch.sin(rad)
    new_y = p[:, 1] * torch.sin(rad) + p[:, 0] * torch.cos(rad)

    return torch.stack([new_y + center, new_x + center], dim=1)


def plot_position_comparison(positions, predictions, file_path):
    """Plot comparison between predicted and real particle positions."""
    plt.figure(figsize=(14, 8))
    grid = plt.GridSpec(4, 7, wspace=0.2, hspace=0.1)

    plt.subplot(grid[1:, :3])
    plt.scatter(positions[:, 0], predictions[:, 0], alpha=0.5)
    plt.axline((25, 25), slope=1, color="black")
    plt.xlabel("True Horizontal Position", fontsize=20)
    plt.ylabel("Predicted Horizontal Position", fontsize=20)
    plt.axis("equal")

    plt.subplot(grid[1:, 4:])
    plt.scatter(positions[:, 1], predictions[:, 1], alpha=0.5)
    plt.axline((25, 25), slope=1, color="black")
    plt.xlabel("True Vertical Position", fontsize=20)
    plt.ylabel("Predicted Vertical Position", fontsize=20)
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(file_path)


def main():
    image_size = 51

    particle = dt.PointParticle(
        position=lambda: uniform(image_size / 2 - 5, image_size / 2 + 5, size=2),
    )

    optics = dt.Fluorescence(output_region=(0, 0, image_size, image_size))
    simulation = (
        optics(particle)
        >> dt.NormalizeMinMax()
        >> dt.Gaussian(sigma=0.1)
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )

    train_dataset = dt.pytorch.Dataset(simulation, length=100)
    test_dataset = dt.pytorch.Dataset(simulation & particle.position, length=5000)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axs):
        image, position = test_dataset[i]
        ax.imshow(image[0], cmap="gray", origin="lower")
        ax.scatter(position[1], position[0], c="r")
        if i != 0:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig("../figures/problem1/particle_plots.png")

    backbone = dl.ConvolutionalNeuralNetwork(
        in_channels=1,
        hidden_channels=[16, 32, 64],
        out_channels=128,
        pool=torch.nn.MaxPool2d(2),
    )
    model = dl.Sequential(backbone, torch.nn.Flatten(), torch.nn.LazyLinear(2))

    localizer = ParticleLocalizer(
        model,
        image_size,
        n_transforms=8,
        loss=torch.nn.L1Loss(),
        optimizer=dl.Adam(lr=1e-3),
    ).create()

    dataloader = dl.DataLoader(train_dataset, batch_size=1, shuffle=True)
    trainer = dl.Trainer(max_epochs=100, accelerator="gpu")

    trainer.fit(localizer, dataloader)

    images, positions = zip(*test_dataset)
    images, positions = torch.stack(images), torch.stack(positions)

    predictions_continous = localizer(images).detach().numpy()

    plot_position_comparison(
        positions,
        predictions_continous,
        "../figures/problem1/predictions_continous.png",
    )
    localizer_with_flips = ParticleLocalizerWithFlips(
        model,
        image_size,
        n_transforms=8,
        loss=torch.nn.L1Loss(),
        optimizer=dl.Adam(lr=1e-3),
    ).create()
    trainer_with_flips = dl.Trainer(max_epochs=100)
    trainer_with_flips.fit(localizer_with_flips, dataloader)

    predictions_discrete = (
        localizer_with_flips(images).detach().numpy() + (image_size - 1) / 2
    )

    plot_position_comparison(
        positions, predictions_discrete, "../figures/problem1/predictions_discrete.png"
    )
    abs_diff = np.abs(predictions_continous - predictions_discrete)
    print(abs_diff.mean())
    # 0.21672666
    print(abs_diff.std())
    # 0.12620191


if __name__ == "__main__":
    main()
