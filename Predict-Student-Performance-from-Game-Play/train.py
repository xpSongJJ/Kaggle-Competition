import time
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from utils import TrainProcessing, MyDataset
import torch.utils.data
from model import HFAttention as create_model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def criterion(y_pred, y_true):
    # y_pred shape: (batch_size, num_classes)
    # y_true shape: (batch_size, num_classes)
    y_true = y_true.float()  # y_true was logit
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


def batch_acc(y_pred, y_true):
    # Sigmoid function maps predicted values to a range of 0 to 1
    y_true = np.asarray(y_true)
    y_pred = torch.sigmoid(y_pred)
    y_pred_logit = np.where(y_pred >= 0.5, 1, 0)
    acc = np.sum(y_pred_logit == y_true) / y_pred_logit.size
    return acc


if os.path.exists("./checkpoint") is False:
    os.makedirs("./checkpoint")
epochs = 10
batch_size = 256
in_len = 96  # sequence length of input which equal to token number

out_len = 18  # total 18 binary category
token_len = 16  # token length, 16 categories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
data_root = './data/'

net = create_model(in_len=in_len, token_len=token_len, d_model=512)
if os.path.exists("./checkpoint/weights.pth") and False:
    net.load_state_dict(torch.load("./checkpoint/weights.pth"))
net.to(device)

df = pd.read_csv(data_root + 'train.csv')
df_proc = TrainProcessing(df)
df_proc.drop_loc(min_len=in_len)
vars_dict = df_proc.get_var(write=True)
level_group_q = {0: [0, 1, 2],
                 1: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 2: [13, 14, 15, 16, 17]}
level_groups = [0, 1, 2]
for level_group in level_groups:
    freeze_layer = {0: [], 1: [net.encoder], 2: [net.encoder, net.encoder2]}
    for fl in freeze_layer[level_group]:
        for param in fl.parameters():
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

    writer = SummaryWriter(log_dir='./runs/' + time.strftime('%m-%d_%H.%M', time.localtime()) + '_' + str(level_group))

    level_data = df_proc.get_data(level_group)
    labels = pd.read_csv('./data/train_labels.csv')
    train_ds = MyDataset(dataframe=level_data, labels=labels, flag='train', seq_len=in_len)
    val_ds = MyDataset(dataframe=level_data, labels=labels, flag='val', seq_len=in_len)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=batch_size,
                                             shuffle=True)
    best_acc = 0
    for epoch in range(epochs):
        # train
        net.train()
        train_loss = []
        train_acc = []
        train_loader = tqdm(train_loader, desc=f"train epoch: {epoch}/{epochs}", colour='green', file=sys.stdout)
        for seqs, labels, stamps in train_loader:
            out = net(seqs.to(device), stamps.to(device))[level_group]
            label = labels[:, level_group_q[level_group]]

            loss = criterion(out, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(batch_acc(out.cpu(), label))
            train_loader.set_postfix({'loss': np.average(train_loss), 'acc': np.average(train_acc)})  # dict

        # evaluate
        net.eval()
        val_loss = []
        val_acc = []
        val_loader = tqdm(val_loader, desc=f"evaluate epoch: {epoch}/{epochs}", file=sys.stdout)
        for seqs, labels, stamps in val_loader:
            out = net(seqs.to(device), stamps.to(device))[level_group]
            label = labels[:, level_group_q[level_group]]

            loss = criterion(out, label.to(device))
            val_loss.append(loss.item())
            val_acc.append(batch_acc(out.cpu(), label))  # build-in sigmoid
            val_loader.set_postfix({'loss': np.average(val_loss), 'acc': np.average(val_acc)})

        val_loss_ep = np.average(val_loss)
        val_acc_ep = np.average(val_acc)
        writer.add_scalar("evaluate_loss", val_loss_ep, epoch)
        writer.add_scalar("evaluate_acc", val_acc_ep, epoch)

        if best_acc < val_acc_ep:
            torch.save(net.state_dict(), "./checkpoint/weights.pth")
            best_acc = val_acc_ep

print("training finished !")
