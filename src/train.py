import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import MoADataset, TestMoADataset
from models import SimpleNet
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from preprocess import prepare_data
from pytorch_tabnet.tab_model import TabNetRegressor
from metrics import LogitsLogLoss
from torch.utils.data import DataLoader
from utils import seed_everything, sigmoid
from scipy.special import expit
from sklearn.metrics import log_loss


EPOCHS = 30
TAB_EPOCHS = 100
BATCH_SIZE = 2048
TAB_BATCH_SIZE = 1042
LEARNING_RATE = 1e-3
TAB_LEARNING_RATE = 2e-2
WEIGHT_DECAY = 2e-5
SEED = 42
NUM_FOLDS = 10


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
    Y_pred_lst = []
    model.eval()
    running_loss = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        loss = loss_fun(outputs, targets)
        running_loss += loss.item()
        Y_pred_lst.append(outputs.sigmoid().detach().cpu().numpy())

        if (batch_idx + 1) % 1024 == 0 or (batch_idx + 1) == len(val_loader):
            print(f'Validate Epoch: {epoch + 1}, Batch: [{batch_idx + 1}/{len(val_loader)}], Loss: {loss.item():.6f}')

    mean_loss = running_loss / len(val_loader)
    Y_pred = np.concatenate(Y_pred_lst)

    return mean_loss, Y_pred


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


def train_simple_net(fold, X_test, train_loader, val_loader, train_size, val_idx, in_size, out_size,
                     device):
    model = SimpleNet(in_size, 2048, 1024, out_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fun = nn.BCEWithLogitsLoss()
    best_loss = np.inf
    oof = np.zeros((train_size, out_size))

    for epoch in range(EPOCHS):
        epoch_train_loss = train_fun(model, optimizer, loss_fun, train_loader, device, epoch)
        epoch_val_loss, val_Y_pred = validate_fun(model, loss_fun, val_loader, device, epoch)

        # print(f'Epoch: {epoch}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}')

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            oof[val_idx] = val_Y_pred
            torch.save(model.state_dict(), f'../fold_models/simple_fold_{fold + 1}.pth')
    
        test_dataset = TestMoADataset(X_test)   
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleNet(in_size, 2048, 1024, out_size).to(device)
    model.load_state_dict(torch.load(f'../fold_models/simple_fold_{fold + 1}.pth'))
    model.to(device)

    Y_pred = test_fun(model, test_loader, device)

    return best_loss, Y_pred, oof


def train_tab_net(fold, X_test, X_train, Y_train, X_val, Y_val, train_size, val_idx, out_size):
    oof = np.zeros((train_size, out_size))
    tabnet_params = dict(
                        n_d=32,
                        n_a=32,
                        n_steps=1,
                        gamma=1.7,
                        lambda_sparse=0,
                        optimizer_fn=torch.optim.Adam,
                        optimizer_params=dict(lr=TAB_LEARNING_RATE, weight_decay=WEIGHT_DECAY),
                        scheduler_params=dict(mode='min',
                                                patience=5,
                                                min_lr=1e-5,
                                                factor=0.9,),
                        scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau,                    
                        mask_type='entmax',
                        seed=SEED,
                        verbose=10
                    )

    model = TabNetRegressor(**tabnet_params)

    model.fit(
        X_train=X_train, 
        y_train=Y_train,
        eval_set = [(X_val, Y_val)],
        eval_name=['val'],
        eval_metric=[LogitsLogLoss],
        max_epochs=TAB_EPOCHS,
        patience=20,
        batch_size=TAB_BATCH_SIZE,
        virtual_batch_size=128,
        num_workers=1,
        drop_last=False,
        loss_fn=nn.BCEWithLogitsLoss()
    )

    model.save_model(f'../fold_models/tabnet_fold_{fold + 1}.pth')

    oof[val_idx] = expit(model.predict(X_val))
    Y_pred = expit(model.predict(X_test))

    return np.min(model.history['val_logits_ll']), Y_pred, oof


def run_msk_fold_cv(X_train, Y_train, Y_train_stub, X_test, ss, num_folds, model_name, device):
    running_loss = 0
    Y_pred = np.zeros((X_test.shape[0], Y_train.shape[1] - 1))
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)
    oof = np.zeros((X_train.shape[0], Y_train.shape[1] - 1))

    for fold, (trn_idx, val_idx) in enumerate(mskf.split(X_train, Y_train)):
        fold_X_train = X_train.loc[trn_idx, :]
        fold_Y_train = Y_train.loc[trn_idx, :].drop('sig_id', axis=1)
        fold_X_val = X_train.loc[val_idx, :]
        fold_Y_val = Y_train.loc[val_idx, :].drop('sig_id', axis=1)

        train_dataset = MoADataset(fold_X_train, fold_Y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MoADataset(fold_X_val, fold_Y_val)   
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f'Fold: {fold + 1}')

        if model_name == 'simple':
            fold_loss, fold_Y_pred, fold_oof = train_simple_net(fold, X_test.drop('sig_id', axis=1), 
                                                                train_loader, val_loader, 
                                                                X_train.shape[0], val_idx, 
                                                                X_train.shape[1], Y_train.shape[1] - 1, 
                                                                device)
        elif model_name == 'tab':
            fold_X_train = fold_X_train.to_numpy()
            fold_Y_train = fold_Y_train.to_numpy()
            fold_X_val = fold_X_val.to_numpy()
            fold_Y_val = fold_Y_val.to_numpy()
            X_test = X_test.to_numpy()
            fold_loss, fold_Y_pred, fold_oof = train_tab_net(fold, X_test.drop('sig_id', axis=1), 
                                                             fold_X_train, fold_Y_train,
                                                             fold_X_val, fold_Y_val,
                                                             X_train.shape[0], val_idx, 
                                                             Y_train.shape[1] - 1)
        Y_pred += fold_Y_pred
        oof += fold_oof
        running_loss += fold_loss

    Y_pred /= num_folds
    oof /= num_folds
    cv_loss = running_loss / num_folds

    oof_Y_pred = Y_train.copy()
    oof_Y_pred.iloc[:, 1:] = oof
    oof_Y_pred = Y_train_stub.loc[:, ['sig_id']].merge(oof_Y_pred, on='sig_id', how='left').fillna(0)

    Y_true = Y_train_stub.iloc[:, 1:].values
    oof_Y_pred = oof_Y_pred.iloc[:, 1:].values

    cv_score = 0

    for i in range(oof_Y_pred.shape[1]):
        cv_score += log_loss(Y_true[:, i], oof_Y_pred[:, i])

    cv_score /= oof_Y_pred.shape[1]

    print(f'CV loss (ctl_vechile excluded): {cv_loss:.6f}')
    print(f'CV loss: {cv_score:.6f}')

    test_Y_pred = X_test.loc[:, ['sig_id']].merge(ss, how='left', on=['sig_id'])
    test_Y_pred.iloc[:, 1:] = Y_pred
    test_Y_pred = ss.loc[:, ['sig_id']].merge(test_Y_pred, on='sig_id', how='left').fillna(0)

    return test_Y_pred


def run_net(model_name, mode):
    use_cuda = False
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    Y_pred = None

    if device == ('cuda'):
        use_cuda = True

    seed_everything(SEED, use_cuda)

    X_train, Y_train, Y_train_stub, X_test, ss = prepare_data('../data/lish-moa')

    if mode == 'cv':
        Y_pred = run_msk_fold_cv(X_train, Y_train, Y_train_stub, X_test, ss, NUM_FOLDS, model_name, device)
  
    return Y_pred
