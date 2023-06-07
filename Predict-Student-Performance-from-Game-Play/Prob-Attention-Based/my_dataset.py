import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataframe, targets: pd.DataFrame, session_ids: list, seq_len: int, use_cols: list):
        self.df = dataframe
        self.targets = targets
        self.seq_len = seq_len
        self.use_cols = use_cols
        self.sess_id = session_ids
        self.groups = self.df.groupby('session_id')

    def __getitem__(self, item):
        session_id = self.sess_id[item]
        data_y = self.targets.loc[session_id].astype(int).values

        data_x = self.groups.get_group(session_id).copy()
        data_x['elapsed_time_diff'] = data_x['elapsed_time'].diff().apply(lambda x: np.log(1+x/1000))
        data_x['elapsed_time'] = data_x['elapsed_time'].apply(lambda x: np.log(1+x/1000))
        data_x = data_x[self.use_cols].fillna(0).values.copy()
        if data_x.shape[0] < self.seq_len:
            pad_len = self.seq_len - data_x.shape[0]
            data_x = np.pad(data_x, ((pad_len, 0), (0, 0)))
        else:
            data_x = data_x[-self.seq_len:, ...]
        assert data_x.shape[0] == self.seq_len, "数据长度不准确"

        stamp = [int(str(session_id)[:2]), int(str(session_id)[2:4])+1, int(str(session_id)[4:6]), int(str(session_id)[6:8])]
        stamp = torch.tensor(stamp, dtype=torch.float32)
        return torch.tensor(data_x, dtype=torch.float32), stamp, torch.tensor(data_y, dtype=torch.float32)

    def __len__(self):
        return len(self.sess_id)
