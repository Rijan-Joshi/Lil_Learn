# Imports

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


import torch
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, latent_dim):
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
    def __init__(self, image_dim):
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


# Optimizers

optimizer = torch.optim.Adam()
