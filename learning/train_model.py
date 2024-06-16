import torch
from torch import nn
from convNeuralNetwork import model as conv_model
from preprocess import train_dataloader
import torch.optim as optim
import joblib


learning_rate = 1e-3
batch_size = 64
epochs = 10
loss_train_fn = nn.CrossEntropyLoss()
train_optimizer = optim.Adam(conv_model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.transpose(1, 2), y)  # Ajustement pour correspondre aux dimensions attendues par CrossEntropyLoss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred.transpose(1, 2), y).item()
            correct += (pred.argmax(2) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size * y.shape[1]
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, conv_model, loss_train_fn, train_optimizer)
    test_loop(train_dataloader, conv_model, loss_train_fn)
print("Done!")
