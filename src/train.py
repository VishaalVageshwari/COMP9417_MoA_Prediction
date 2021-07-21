import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import MoADataset, TestMoADataset
from models import SimpleNet
from preprocess import prepare_data
from torch.utils.data import DataLoader
from utils import seed_everything
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


EPOCHS = 30
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
NUM_FOLDS = 5


def train_fun(model, optimizer, loss_fun, train_loader, device, epoch):
    model.train()
    running_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fun(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'Train Epoch: {epoch + 1}, Batch: [{(batch_idx + 1)}/{len(train_loader)}], Loss: {loss.item():.3f}')

    mean_loss = running_loss / len(train_loader)
    return mean_loss


def validate_fun(model, loss_fun, val_loader, device, epoch):
    model.eval()
    running_loss = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        loss = loss_fun(outputs, targets)
        running_loss += loss.item()

        if (batch_idx + 1) % 1024 == 0 or (batch_idx + 1) == len(val_loader):
            print(f'Validate Epoch: {epoch + 1}, Batch: [{batch_idx + 1}/{len(val_loader)}], Loss: {loss.item():.6f}')

    mean_loss = running_loss / len(val_loader)
    return mean_loss


def test_fun(model, test_loader, device):
    Y_pred_lst = []
    model.eval()

    for batch_idx, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)

        Y_pred_lst.append(outputs.sigmoid().detach().cpu().numpy())

    Y_pred = np.concatenate(Y_pred_lst)

    return Y_pred


def train_simple_net(fold, X_test, train_loader, val_loader, in_size, out_size, device):
    model = SimpleNet(in_size, 2048, 1024, out_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fun = nn.BCEWithLogitsLoss()
    best_loss = np.inf

    for epoch in range(EPOCHS):
        epoch_train_loss = train_fun(model, optimizer, loss_fun, train_loader, device, epoch)
        epoch_val_loss = validate_fun(model, loss_fun, val_loader, device, epoch)

        # print(f'Epoch: {epoch}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}')

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), f'../fold_models/simple_fold_{fold + 1}.pth')
    
        test_dataset = TestMoADataset(X_test)   
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleNet(in_size, 2048, 1024, out_size).to(device)
    model.load_state_dict(torch.load(f'../fold_models/simple_fold_{fold + 1}.pth'))
    model.to(device)

    Y_pred = test_fun(model, test_loader, device)

    return best_loss, Y_pred


def run_msk_fold_cv(X_train, Y_train, X_test, num_folds, model_name, device):
    running_loss = 0
    Y_pred = np.zeros((X_test.shape[0], Y_train.shape[1]))
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)

    for fold, (trn_idx, val_idx) in enumerate(mskf.split(X_train, Y_train)):
        fold_X_train = X_train.loc[trn_idx, :]
        fold_Y_train = Y_train.loc[trn_idx, :]
        fold_X_val = X_train.loc[val_idx, :]
        fold_Y_val = Y_train.loc[val_idx, :]

        train_dataset = MoADataset(fold_X_train, fold_Y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MoADataset(fold_X_val, fold_Y_val)   
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f'Fold: {fold + 1}')

        if model_name == 'simple':
            fold_loss, fold_Y_pred = train_simple_net(fold, X_test, train_loader, val_loader, X_train.shape[1], Y_train.shape[1], device)
            Y_pred += fold_Y_pred
            running_loss += fold_loss

    Y_pred /= num_folds
    cv_loss = running_loss / num_folds

    print(f'CV loss: {cv_loss:.6f}')

    return Y_pred


def run_simple_net(mode):
    use_cuda = False
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    Y_pred = None

    if device == ('cuda'):
        use_cuda = True

    seed_everything(SEED, use_cuda)

    X_train, Y_train, X_test, ss = prepare_data('../data/lish-moa')

    if mode == 'cv':
        Y_pred = run_msk_fold_cv(X_train, Y_train, X_test, NUM_FOLDS, 'simple', device)

    ss.iloc[:, 1:] = Y_pred
    return ss
