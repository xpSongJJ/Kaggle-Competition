import sys
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from utils import MyDataset, BCEFocalLoss, best_f1_threshold, create_lr_scheduler
import torch.utils.data
from tqdm import tqdm
from model import ViT
from config import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Begin training, using device {device}")

targets = pd.read_csv('./data/label_lvgp.csv')
target_groups = targets.groupby('level_group')
with open('./data/positive_duty.json', 'r') as f:
    positive_duty = json.load(f)

df = pd.read_parquet('../data/train.parquet', columns=read_cols)
for cat in cat2codes:
    df[cat] = df[cat].map(cat2codes[cat])
df_groups = df.groupby('level_group')

session_n = df['session_id'].nunique()
random.seed(42)
idx = random.sample(range(session_n), session_n)
split = round(0.8*session_n)
tr_idx = idx[:split]
val_idx = idx[session_n-split:]

for lvgp in LEVEL_GROUP:
    train_val_df = df_groups.get_group(lvgp).reset_index(drop=True)
    train_val_label = target_groups.get_group(lvgp).reset_index(drop=True)
    train_ds = MyDataset(dataframe=train_val_df, targets=train_val_label, index=tr_idx, seq_len=SEQ_LEN, use_cols=use_cols)
    val_ds = MyDataset(dataframe=train_val_df, targets=train_val_label, index=val_idx, seq_len=SEQ_LEN, use_cols=use_cols)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=BATH_SIZE,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=BATH_SIZE,
                                             shuffle=True)

    # criterion = torch.nn.BCELoss()
    m = torch.nn.Sigmoid()

    for i, qst in enumerate(LEVEL_GROUP_QST[lvgp]):
        net = ViT(num_classes=1, token_len=TOKEN_SIZE, seq_len=SEQ_LEN)
        net.to(device)
        if os.path.exists(f"./checkpoint/weights_{qst}.pth") and False:
            net.load_state_dict(torch.load(f"./checkpoint/weights_{qst}.pth"))

        al = 1 - positive_duty[str(qst)]
        criterion = BCEFocalLoss(gamma=2, alpha=al)
        best_f1 = 0
        best_thred = 0
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.1)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), EPOCH, warmup=True, warmup_epochs=1)
        for epoch in range(EPOCH):
            # train
            net.train()
            train_loss = []

            train_loader = tqdm(train_loader, desc=f"Question {qst} train epoch: {epoch}/{EPOCH}",
                                colour='green', file=sys.stdout)
            for seqs, stamp, labels in train_loader:
                out = net(seqs.to(device), stamp.to(device))
                label = labels[:, i]

                loss = criterion(m(out), label.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss.append(loss.item())
                lr_ = optimizer.state_dict()['param_groups'][0]['lr']
                train_loader.set_postfix({'loss': np.average(train_loss), 'lr': lr_})  # dict

            # evaluate
            net.eval()
            val_loss = []
            y_pred = []
            y_true = []
            val_loader = tqdm(val_loader, desc=f"question {qst} evaluate epoch: {epoch}/{EPOCH}",
                              file=sys.stdout)
            with torch.no_grad():
                for seqs, stamp, labels in val_loader:
                    out = net(seqs.to(device), stamp.to(device))
                    label = labels[:, i]

                    loss = criterion(m(out), label.to(device))

                    val_loss.append(loss.item())
                    y_pred.extend(list(out.cpu().flatten()))
                    label = np.asarray(label)
                    y_true.extend(list(label.flatten()))

                    val_loader.set_postfix({'loss': np.average(val_loss)})

            val_loss_ep = np.average(val_loss)
            f1, thred = best_f1_threshold(y_true, y_pred, epoch, i)
            print(f'-------------f1_score: {f1}, threshold: {thred}---------------')

            if best_f1 < f1:
                torch.save(net.state_dict(), f"./checkpoints/weights_{qst}.pth")
                best_f1 = f1
                best_thred = thred

print("Training finished !")
