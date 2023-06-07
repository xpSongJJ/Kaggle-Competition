import time
import sys
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from utils import reduce_memory_usage, best_f1_threshold
import torch.utils.data
from model import HFAttention as create_model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from my_dataset import MyDataset
from config import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Begin training using device: {device} ...")

criterion = torch.nn.BCEWithLogitsLoss()

net = create_model(seq_len=SEQ_LEN, token_size=TOKEN_SIZE, d_model=256)
if os.path.exists(f"./checkpoint/weights.pth"):
    net.load_state_dict(torch.load(f"./checkpoint/weights.pth"))
net.to(device)

targets = pd.read_parquet(DATA_ROOT+"labels_all.parquet")

df = pd.read_parquet(DATA_ROOT+"train.parquet", columns=read_cols)
for cat in cat2codes:
    df[cat] = df[cat].map(cat2codes[cat])
df_groups = df.groupby('level_group')

session_id = df['session_id'].unique()
session_n = len(session_id)
random.seed(42)
idx = random.sample(range(session_n), session_n)
split = round(0.8*session_n)
tr_idx = idx[:split]
val_idx = idx[session_n-split:]
tr_sess_id = [session_id[i] for i in tr_idx]
val_sess_id = [session_id[i] for i in val_idx]


for step, level_group in enumerate(LEVEL_GROUP):
    freeze_layer = {'0-4': [], '5-12': [net.encoder], '13-22': [net.encoder, net.encoder2]}
    for fl in freeze_layer[level_group]:
        for param in fl.parameters():
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=5e-4)

    writer = SummaryWriter(log_dir='./runs/' + time.strftime('%m-%d_%H.%M', time.localtime()) + '_' + str(level_group))

    train_val_df = df_groups.get_group(level_group).reset_index(drop=True)
    train_ds = MyDataset(dataframe=train_val_df, targets=targets, session_ids=tr_sess_id, seq_len=SEQ_LEN,
                         use_cols=use_cols)
    val_ds = MyDataset(dataframe=train_val_df, targets=targets, session_ids=val_sess_id, seq_len=SEQ_LEN,
                       use_cols=use_cols)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)
    best_f1 = 0
    for epoch in range(EPOCH):
        # train
        net.train()
        train_pred = []
        train_target = []
        train_loss = []

        train_loader = tqdm(train_loader, desc=f"train epoch: {epoch}/{EPOCH}",
                            colour=['green', 'blue', 'cyan'][step], file=sys.stdout)
        for seqs, stamp, labels in train_loader:
            out = net(seqs.to(device), stamp.to(device))[step]
            label = labels[:, LEVEL_GROUP_QST[level_group]]

            loss = criterion(out, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pred.extend(list(out.cpu().flatten()))
            train_target.extend(list(label.flatten()))
            train_loss.append(loss.item())
            train_loader.set_postfix({'loss': np.average(train_loss)})

        f1, thred = best_f1_threshold(train_target, train_pred)
        print(f"training best f1: {f1}, best threshold: {thred} !")
        # evaluate
        net.eval()
        val_loss = []
        y_pred = []
        y_true = []
        val_loader = tqdm(val_loader, desc=f"evaluate epoch: {epoch}/{EPOCH}", file=sys.stdout)
        with torch.no_grad():
            for seqs, stamp, labels in val_loader:
                out = net(seqs.to(device), stamp.to(device))[step]
                label = labels[:, LEVEL_GROUP_QST[level_group]]

                loss = criterion(out, label.to(device))
                val_loss.append(loss.item())

                y_true.extend(list(label.flatten()))
                y_pred.extend(list(out.cpu().flatten()))

                val_loader.set_postfix({'loss': np.average(val_loss)})

        val_loss_ep = np.average(val_loss)
        f1, thred = best_f1_threshold(y_true, y_pred)
        print(f'Evaluating best f1_score: {f1}, best threshold: {thred} !')
        writer.add_scalar("evaluate_loss", val_loss_ep, epoch)
        writer.add_scalar("evaluate_acc", f1, epoch)

        if best_f1 < f1:
            torch.save(net.state_dict(), "./checkpoint/weights.pth")
            best_f1 = f1

print("training finished !")
