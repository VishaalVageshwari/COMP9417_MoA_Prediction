import torch
import torch.nn as nn
import torch.optim as optim

from datasets import MoADataset
from models import SimpleNet
from preprocess import prepare_data
from torch.utils.data import DataLoader
from utils import seed_everything


EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


def train(model, device, loss_fun, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fun(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Train Epoch: {epoch + 1}\tLoss: {loss.item():.6f}')


def train_simple_net():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(42)

    X_train, X_test, Y_train = prepare_data('../data/lish-moa')
    print(X_train.shape)
    print(Y_train.shape)

    train_dataset = MoADataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleNet(X_train.shape[1], 2048, 1024, Y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fun = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        train(model, device, loss_fun, train_loader, optimizer, epoch)

