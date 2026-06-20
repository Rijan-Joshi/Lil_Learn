# Imports
import os

from numpy.random import random_sample
from sympy import discriminant
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


import torch
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision.utils import save_image
import torchvision.utils as vutils

import torchvision
from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device used is " + device)

torch.manual_seed(100)
np.random.seed(100)


# Loading the MNIST datasets

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        # transforms.Lambda(
        #     lambda x: x.view(-1)
        # ),  # Transforms Image from (28, 28) to (784)
    ]
)

train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


# Model for Generators and Discriminators


# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, X):
        X = self.model(X)
        X = X.view(-1, 1, 28, 28)
        return X


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, image_dim=784):
        super(Discriminator, self).__init__()
        self.image_dim = image_dim

        self.model = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output probability
        )

    def forward(self, X):
        X = X.view(-1, self.image_dim)
        return self.model(X)


# Connecting models to cuda if available
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers and Loss Functions
lr = 0.0002
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

# Image Plotting Utitlity: create random noise, feed the generator, generate and then save the image for each epoch
random_dim = 100


def save_generated_images(generator, random_examples, epoch):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(random_examples, random_dim, device=device)
        generated = generator(noise)  # Returns 28 * 28 images
        # generated = (generated + 1) / 2  # Changing pixel values from [-1, 1] to [0,1]

        os.makedirs("generated_images", exist_ok=True)

        # grid = vutils.make_grid(generated, nrow=10, normalize=True)
        # grid_np = grid.cpu().numpy().transpose((1, 2, 0))

        # plt.figure(figsize=(10, 10))
        # plt.axis("off")
        # plt.title("Generated Digits")
        # plt.imshow(grid_np)
        # plt.show()

        save_image(
            generated,
            f"generated_images/images_epoch_{epoch}.png",
            nrow=10,
            normalize=True,
        )

    generator.train()


# Training Function: generate fake images using generator and train on both real and fake images by adding up the loss values
epochs = 400


def train():
    for epoch in range(1, epochs + 1):

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch [{epoch:2d}/ {epochs:2d}]",
            leave=True,
        )

        for batch_idx, (real_images, _) in progress_bar:

            batch_size = real_images.size(0)

            # Real Images and real labels i.e. 1
            real_images = real_images.to(device)
            real_labels = torch.full((batch_size, 1), 0.9).to(device)

            # Fake Images and fake labels i.e. 0
            latent_dim = 100
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()

            # Loss on real images
            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, real_labels)

            # Calculate loss on fake images
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            # Total Loss
            total_d_loss = real_loss + fake_loss

            # Backward Pass
            total_d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            g_optimizer.step()

            progress_bar.set_postfix(
                D_loss=f"{total_d_loss.item():.4f}", G_loss=f"{g_loss.item():.4f}"
            )

        if epoch % 5 == 0:
            save_generated_images(generator, 100, epoch)

    print("Training complted")


if __name__ == "__main__":
    train()
