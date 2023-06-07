import json
import os
import math

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt)\
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class MyDataset(Dataset):
    def __init__(self, dataframe, targets: pd.DataFrame, index: list, seq_len: int, use_cols: list):
        self.df = dataframe
        self.targets = targets
        self.seq_len = seq_len
        self.use_cols = use_cols
        self.index = index
        self.groups = self.df.groupby('session_id')

    def __getitem__(self, item):
        idx = self.index[item]
        session_id = self.targets.loc[idx, 'session_id']
        data_y = eval(self.targets.loc[idx, 'correct'])

        data_x = self.groups.get_group(session_id).copy()
        data_x['elapsed_time_diff'] = data_x['elapsed_time'].diff().apply(lambda x: np.log(1+x/1000))
        data_x['elapsed_time'] = data_x['elapsed_time'].apply(lambda x: np.log(1+x/1000))
        data_x = data_x[self.use_cols].fillna(0).values

        if data_x.shape[0] < self.seq_len:
            pad_len = self.seq_len - data_x.shape[0]
            data_x = np.pad(data_x, ((pad_len, 0), (0, 0)))
        else:
            data_x = data_x[-self.seq_len:, ...]
        assert data_x.shape[0] == self.seq_len, "数据长度不准确"

        stamp = [int(str(session_id)[:2]), int(str(session_id)[2:4])+1, int(str(session_id)[4:6]), int(str(session_id)[6:8])]
        stamp = torch.tensor(stamp, dtype=torch.int32)
        return torch.tensor(data_x, dtype=torch.float32), stamp, torch.tensor(data_y, dtype=torch.float32)

    def __len__(self):
        return len(self.index)


def reduce_memory_usage(df):
    if isinstance(df, pd.DataFrame):
        start_mem = df.memory_usage(index=True, deep=True).sum() / 1024 ** 2
        print(f'Initial memory usage of dataframe is {round(start_mem, 2)} MB')
        for col in df.columns:
            col_type = df[col].dtype.name
            if (col_type != 'datetime64[ns]') & (col_type != 'category'):
                if col_type != 'object':
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            pass
                else:
                    df[col] = df[col].astype('category')
        mem_usg = df.memory_usage(index=True, deep=True).sum() / 1024 ** 2
        print(f"Memory usage of became: {round(mem_usg, 2)} MB")
        return df


def best_f1_threshold(targets: list, pred: list, epoch: int, qst: int):
    best_threshold = None

    filepath = f"./log/f1_threshold_q{qst}.json"
    if not os.path.isfile(filepath):
        with open(filepath, 'w') as f:
            json.dump({}, f)
    with open(filepath, 'r') as f:
        f1_threshold = json.load(f)

    f1_thred = {}
    best_f1 = 0
    for thred in np.around(np.arange(0., 0.8, 0.02), decimals=2):
        pred_logit = [1 if torch.sigmoid(p) >= thred else 0 for p in pred]
        f1 = f1_score(targets, pred_logit, average='macro')
        f1_thred[thred] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thred
    f1_threshold[epoch] = f1_thred
    with open(filepath, 'w') as f:
        json.dump(f1_threshold, f)
    return best_f1, best_threshold


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    """params
    num_step: batch number
    epochs: epoch number
    warmup_epoch: epoch number in which warmup implementing
    warmup_factor: lr*warnup_factor while start-upping
    end_factor: lr while ending
    """
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """params
        x: x++ once .step() be called
        """
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
