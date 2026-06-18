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
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, vutils

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
    root="./data", train=False, download=True, transform=transform
)

test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

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
    with generator.no_eval():
        noise = torch.randn(random_examples, random_dim, device=device)
        generated = generator(noise)  # Returns 28 * 28 images
        generated = (generated + 1) / 2  # Changing pixel values from [-1, 1] to [0,1]

        os.makedirs("generated_images", exist_ok=True)

        grid = vutils.make_grid(generated, nrow=10, normalize=True)
        grid_np = grid.cpu().numpy().transpose((1, 2, 0))

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.title("Generated Digits")
        plt.imshow(grid_np)
        plt.show()

        save_image(generated, f"generated_images/images_epoch_{epoch}.png", nrow=10)

    generator.train()


# Training Function


def train(): ...
