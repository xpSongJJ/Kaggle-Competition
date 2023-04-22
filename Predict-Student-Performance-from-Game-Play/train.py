import time
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from utils import MyDataset, reduce_memory_usage
import torch.utils.data
from net import HFAttention as create_model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def criterion(y_pred, y_true):
    # y_pred shape: (batch_size, num_classes)
    # y_true shape: (batch_size, num_classes)
    y_true = y_true.float()  # y_true was logit
    y_pred = torch.squeeze(y_pred)
    y_true = torch.squeeze(y_true)
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


def batch_acc(y_pred, y_true):
    # Sigmoid function maps predicted values to a range of 0 to 1
    y_true = np.asarray(y_true)
    y_pred = torch.sigmoid(y_pred)
    y_pred_logit = np.where(y_pred >= 0.5, 1, 0)
    acc = np.sum(y_pred_logit == y_true) / y_pred_logit.size
    return acc


config = {
    'epoch': 1,
    'batch_size': 32,
    'device': torch.device("cuda"),
    'lr': 1e-3
}

if os.path.exists("./checkpoint") is False:
    os.makedirs("./checkpoint")

seq_len = 96  # sequence length of input which equal to token number
out_len = 18  # total 18 questions
token_size = 19  # token length, 15 columns
net = create_model(seq_len=seq_len, token_size=token_size, d_model=512)

if os.path.exists("./checkpoint/weights.pth") and False:
    net.load_state_dict(torch.load("./checkpoint/weights.pth"))
device = config['device']
net.to(device)

labels = pd.read_csv('./data/train_labels.csv')
labels = labels.set_index('session_id')['correct'].to_dict()

train_path = {'0-4': './data/train_0-4.csv',
              '5-12': './data/train_5-12.csv',
              '13-22': './data/train_13-22.csv'}

level_group_q = {0: [0, 1, 2],
                 1: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 2: [13, 14, 15, 16, 17]}
level_groups = ['0-4', '5-12', '13-22']
for step, level_group in enumerate(level_groups):
    freeze_layer = {'0-4': [], '5-12': [net.encoder], '13-22': [net.encoder, net.encoder2]}
    for fl in freeze_layer[level_group]:
        for param in fl.parameters():
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

    writer = SummaryWriter(log_dir='./runs/' + time.strftime('%m-%d_%H.%M', time.localtime()) + '_' + str(level_group))

    train_df = pd.read_csv(train_path[level_group])
    train_df = reduce_memory_usage(train_df)
    labels = pd.read_csv('./data/train_labels.csv')
    labels = labels.set_index('session_id')['correct'].to_dict()
    train_ds = MyDataset(dataframe=train_df, labels=labels, flag='train', seq_len=seq_len)
    val_ds = MyDataset(dataframe=train_df, labels=labels, flag='val', seq_len=seq_len)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=config['batch_size'],
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=config['batch_size'],
                                             shuffle=True)
    best_acc = 0
    for epoch in range(config['epoch']):
        # train
        net.train()
        train_loss = []
        train_acc = []

        train_loader = tqdm(train_loader, desc=f"train epoch: {epoch}/{config['epoch']}",
                            colour=['green', 'blue', 'cyan'][step], file=sys.stdout)
        for seqs, labels in train_loader:
            out = net(seqs.to(device))[step]
            label = labels[:, level_group_q[step]]

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
        val_loader = tqdm(val_loader, desc=f"evaluate epoch: {epoch}/{config['epoch']}", file=sys.stdout)
        for seqs, labels in val_loader:
            out = net(seqs.to(device))[step]
            label = labels[:, level_group_q[step]]

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

    del train_ds, train_loader, val_ds, val_loader

print("training finished !")
