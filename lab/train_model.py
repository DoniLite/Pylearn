import torch
from torch import nn
from convNeuralNetwork import model as conv_model, device
from preprocess import train_dataloader
import torch.optim as optim
import joblib


learning_rate = 1e-3
epochs = 10
loss_train_fn = nn.CrossEntropyLoss()
train_optimizer = optim.Adam(conv_model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(x)
        # Reshape pred and y to match expected input for CrossEntropyLoss
        y = y.flatten()
        pred = pred.flatten()
        # print(f"pred shape: {pred.shape}")
        # print(f"y shape: {y.shape}")
        loss = loss_fn(pred, y.float())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # Reshape pred and y to match expected input for CrossEntropyLoss
            pred = pred.flatten()  # Shape: (batch_size * sequence_length, num_classes)
            y = y.flatten()  # Shape: (batch_size * sequence_length)
            test_loss += loss_fn(pred, y.float()).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, conv_model, loss_train_fn, train_optimizer)
        test_loop(train_dataloader, conv_model, loss_train_fn)
    print("Done!")


def save_model():
    torch.save(conv_model.state_dict(), '../models/model_weights.pth')
    torch.save(conv_model, '../models/model.pth')
