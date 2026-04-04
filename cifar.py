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


model_ff = FeedForward().to(device)
model_fft = FeedForwardTanh().to(device)
model_cnn = FeedForwardTanh().to(device)



# TRAIN AND TEST



# train model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# test model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        accuracy = correct / size
        print(f"accuracy: {accuracy}\n")

# RUN
# ------------------
# model 1

#constants 
LOSS_FN = nn.CrossEntropyLoss()
EPOCHS = 35

def run(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, LOSS_FN, optimizer)
        test(test_dataloader, model, LOSS_FN)
    print("Done!")



def main():
    run(model_ff)
    # run(model_fft)
    # run(model_cnn)





if __name__ == "__main__":
    main()






