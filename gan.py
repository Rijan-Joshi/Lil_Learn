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
        transforms.Normalize((0.1307,), (0.3081)),
        transforms.Lambda(
            lambda x: x.view(-1)
        ),  # Transforms Image from (28, 28) to (784)
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
    def __init__(self): ...


# Optimizers

optimizer = torch.optim.Adam()
