import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor



# download training data
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# download test data
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64

# create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# set device for training
device = (
    "cuda" 
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
    

# Define models

# model 1 (feed forward)
class FeedForward (nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential (
            nn.Linear(32*32*3, 256),
            nn.ReLU(),
            nn.Linear(256, 128), # alter
            nn.ReLU(),
            nn.Linear(128, 64), # alter
            nn.ReLU(),
            nn.Linear(64, 10) #alter

        )

    def forward(self, x):
      x = self.flatten(x)
      logits = self.stack(x)
      return logits
    


# model 2 (feed forward with Tanh)
class FeedForwardTanh (nn.Module): # if I change this up enough may need to rename
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential (
            nn.Linear(32*32*3, 256),
            nn.Tanh(),
            nn.Linear(256, 128), # alter
            nn.Tanh(),
            nn.Linear(128, 64), # alter
            nn.Tanh(),
            nn.Linear(64, 10) #alter

        )

    def forward(self, x):
      x = self.flatten(x)
      logits = self.stack(x)
      return logits



# model 3 (convolutional layers)
class ConvolutionalNetwork (nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential (
            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 10) # alter

        )

    def forward(self, x):
        logits = self.stack(x)
        return logits




# continue...



