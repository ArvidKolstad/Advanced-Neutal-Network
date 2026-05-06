from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import time
from datetime import timedelta
from torchvision.datasets import MNIST
import torch
import matplotlib.pyplot as plt
import deeplay as dl
import numpy as np


def load_data():
    transform = Compose(
        [Resize((64, 64)), ToTensor(), Normalize(mean=[0.5], std=[0.5], inplace=True)]
    )
    trainset = MNIST(root="data", train=True, transform=transform, download=True)
    # plot_data_set(trainset)
    return trainset


def get_device():
    """Select device where to perform the computations."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def plot_data_set(trainset):
    fig, axs = plt.subplots(1, 8, figsize=(15, 3))
    for ax in axs.ravel():
        img, label = trainset[torch.randint(0, len(trainset), (1,)).squeeze()]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}", fontsize=16)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("../figures/problem4/plot_data_set.png")


def main():
    trainset = load_data()
    device = get_device()
    print(f"Selected device: {device}")

    latent_dim = 100

    gen = (
        dl.DCGANGenerator(
            latent_dim=latent_dim,
            features_dim=64,
            output_channels=1,
        )
        .build()
        .to(device)
    )

    disc = (
        dl.DCGANDiscriminator(
            input_channels=1,
            features_dim=64,
        )
        .build()
        .to(device)
    )
    loader = dl.DataLoader(trainset, batch_size=128, shuffle=True)
    loss = torch.nn.BCELoss()
    optim_gen = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_disc = torch.optim.Adam(disc.parameters(), lr=0.00001, betas=(0.5, 0.999))

    epochs = 10

    num_batches = len(loader)
    gen_losses_avg, disc_losses_avg = [], []
    fix_latent_vector = torch.randn(30, latent_dim, 1, 1).to(device)
    for epoch in range(epochs):
        gen.train(), disc.train()

        print("\n" + f"Epoch {epoch + 1 }/{epochs}" + "\n" + "-" * 10)
        start_time = time.time()

        running_gen_loss, running_disc_loss = 0.0, 0.0
        for batch_idx, (real_images, class_labels) in enumerate(loader, start=0):
            real_images = real_images.to(device)

            noise = torch.randn(loader.batch_size, latent_dim, 1, 1).to(device)
            fake_images = gen(noise)

            # 1. Discriminator training: minimize - log(D(x)) - log(1 - D(G(z))).
            real_output = disc(real_images).reshape(-1)
            fake_output = disc(fake_images).reshape(-1)

            real_loss = loss(real_output, torch.ones_like(real_output))
            fake_loss = loss(fake_output, torch.zeros_like(fake_output))

            disc_loss = (real_loss + fake_loss) / 2

            optim_disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            optim_disc.step()

            # 2. Generator training: minimize - log(D(G(z))).
            fake_output = disc(fake_images).reshape(-1)
            gen_loss = loss(fake_output, torch.ones_like(fake_output))

            optim_gen.zero_grad()
            gen_loss.backward()
            optim_gen.step()

            if batch_idx % 100 == 0:
                print(
                    f"Batch {batch_idx + 1}/{num_batches}: "
                    f"Generator Loss: {gen_loss.item():.4f}, "
                    f"Discriminator Loss: {disc_loss.item():.4f}"
                )

            running_gen_loss += gen_loss.item()
            running_disc_loss += disc_loss.item()

        gen_losses_avg.append(running_gen_loss / num_batches)
        disc_losses_avg.append(running_disc_loss / num_batches)
        end_time = time.time()

        print(
            "-" * 10 + "\n" + f"Epoch {epoch + 1}/{epochs}: "
            f"Generator Loss: {gen_losses_avg[-1]:.4f}, "
            f"Discriminator Loss: {disc_losses_avg[-1]:.4f}, "
            f"Time taken: {timedelta(seconds=end_time - start_time)}"
        )

        gen.eval(), disc.eval()
        fake_images = gen(fix_latent_vector).detach().cpu().numpy()

        fig, axs = plt.subplots(3, 10, figsize=(20, 6))
        for i, ax in enumerate(axs.ravel()):
            ax.imshow(fake_images[i][0], cmap="gray")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig("../figures/problem4/generated_images3.png")

    # PLOT THE TRAINING LOSSES

    plt.figure()
    plt.plot(
        np.arange(len(gen_losses_avg)), gen_losses_avg, "g--o", label="Generator Loss"
    )
    plt.plot(
        np.arange(len(disc_losses_avg)),
        disc_losses_avg,
        "r-o",
        label="Discriminator Loss",
    )
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend()
    plt.savefig("../figures/problem4/training_loss3.png")


if __name__ == "__main__":
    main()
