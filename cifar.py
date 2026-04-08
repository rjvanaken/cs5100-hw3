import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from config import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# local constant
LOSS_FN = nn.CrossEntropyLoss()

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

batch_size = BATCH_SIZE

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
            nn.Linear(32*32*3, 512),
            nn.Tanh(),
            nn.Linear(512, 256), # alter
            nn.Tanh(),
            nn.Linear(256, 128), # alter
            nn.Tanh(),
            nn.Linear(128, 10) #alter

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
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10) # alter

        )

    def forward(self, x):
        logits = self.stack(x)
        return logits


model_ff = FeedForward().to(device)
model_fft = FeedForwardTanh().to(device)
model_cnn = ConvolutionalNetwork().to(device)



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
        total_loss = test_loss / num_batches

        return (accuracy, total_loss)






# RUN
# ------------------
# model 1

def run(model, title, lr=DEFAULT_LR, epochs=DEFAULT_EPOCHS):
    
    print("-" * 33)
    print(title.upper())
    print(f"{"-" * 33}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr)
    losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n{"-" * 33}")
        train(train_dataloader, model, LOSS_FN, optimizer)
        accuracy, total_loss = test(train_dataloader, model, LOSS_FN)

        if losses:
            if total_loss > losses[-1]:
                print("\n Loss increased, terminating early")
                break
        

        losses.append(total_loss)
        print(f"\naccuracy: {accuracy}")
        print(f"total avg loss: {total_loss}\n\n")
    torch.save(model.state_dict(), f"{title}.pth")
        
    return losses


def plotLossGraph(losses, plot_title, filename):
    plt.clf()
    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses, 'o-r')
    plt.savefig(filename)


def plotExampleImage(image, abrev, example_type):
    img, true_label, pred_label = image
    plt.imshow(img.cpu().permute(1, 2, 0))
    plt.title(f"True: {training_data.classes[true_label]}, Predicted: {training_data.classes[pred_label]}")
    plt.axis('off')
    plt.savefig(f"{abrev}_{example_type}.png")
    plt.clf()


def find_examples(dataloader, model, abrev):
    model.eval()
    correct_img, incorrect_img = None, None
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(1)
            for i in range(len(X)):
                if correct_img is None and pred[i] == y[i]:
                    correct_img = (X[i], y[i], pred[i])
                if incorrect_img is None and pred[i] != y[i]:
                    incorrect_img = (X[i], y[i], pred[i])
                if correct_img and incorrect_img:
                    plotExampleImage(correct_img, abrev, "correct")
                    plotExampleImage(incorrect_img, abrev, "incorrect")
                    return
                

                




def main():
    ff = "Feed Forward"
    fft = "Feed Forward with Tanh"
    cnn = "Convolutional Network"

    # default epochs and learning rate
    ff_losses = run(model_ff, ff)
    fft_losses = run(model_fft, fft)
    cnn_losses = run(model_cnn, cnn)
    
    plotLossGraph(ff_losses, ff, "ff_loss.png")
    plotLossGraph(fft_losses, fft, "fft_loss.png")
    plotLossGraph(cnn_losses, cnn, "cnn_loss.png")


    print ("FINAL TEST RESULTS - ACCURACY BY MODEL:")
    for model, title, abrev in [(model_ff, ff, "ff"), (model_fft, fft, "fft"), (model_cnn, cnn, "cnn")]:
        accuracy, total_loss = test(test_dataloader, model, LOSS_FN)

        print(f"{title}: {(accuracy * 100):.2f}%")
        find_examples(test_dataloader, model, abrev)

    




if __name__ == "__main__":
    main()






