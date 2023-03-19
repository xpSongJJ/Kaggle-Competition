import json
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class TrainProcessing:
    """
    about fill NaN:
    numeric columns: fill nan with mode or 0
    category columns: fill nan with -1
    """

    def __init__(self, DataFrame, mode=None):
        if mode is None:
            mode = {'room_coor_x': 426.725528, 'room_coor_y': -102.,
                    'screen_coor_x': 822., 'screen_coor_y': 431., 'hover_duration': 17.}
        self.df = DataFrame
        self.var = {}
        self.drop_columns()
        self.mode = mode
        self._quantize()
        self._slimming()

    def drop_columns(self, drop_columns=None):
        drop_columns = ['fullscreen', 'hq', 'music'] if drop_columns is None else drop_columns
        for col in drop_columns:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)
        return self

    def _quantize(self):
        # index
        self.df['index'] = self.df.groupby('session_id').cumcount() + 1

        # elapsed_time, convert ms to s by divide by 1000, drop the index whose elapsed_time > 6000
        self.df['elapsed_time'] = self.df['elapsed_time'].apply(lambda x: x / 1000).astype(np.float32)
        self.df.drop(self.df[self.df['elapsed_time'] > 6000].index, inplace=True)
        self.df['index_time'] = self.df.groupby('session_id')['elapsed_time'].diff().fillna(
            self.df['elapsed_time']).astype(np.float32)

        # fill nan with median for numerical columns
        cols_to_fill = ['room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']
        self.df[cols_to_fill] = self.df[cols_to_fill].fillna(self.mode)

        # category map
        # event_name, no NaN
        self.df['event_name'] = self.df['event_name'].astype('category')
        event_name_categories = self.df['event_name'].cat.categories  #
        self.df['event_name'] = self.df['event_name'].cat.codes
        # name, 6 category, no NaN
        self.df['name'] = self.df['name'].astype('category')
        name_categories = self.df['name'].cat.categories  #
        self.df['name'] = self.df['name'].cat.codes
        # level, numeric
        # level_group, 3 category, no NaN
        self.df['level_group'] = pd.Categorical(self.df.level_group, ['0-4', '5-12', '13-22'])
        level_group_categories = self.df['level_group'].cat.categories  #
        self.df['level_group'] = self.df['level_group'].cat.codes
        # room_fqid, no NaN
        self.df['room_fqid'] = self.df['room_fqid'].astype('category')
        room_fqid_categories = self.df['room_fqid'].cat.categories  #
        self.df['room_fqid'] = self.df['room_fqid'].cat.codes
        # page, contain NaN
        self.df['page'] = self.df['page'].astype('category')
        page_categories = self.df['page'].cat.categories  #
        self.df['page'] = self.df['page'].cat.codes
        # fqid, contain NaN, 128 kinds including NaN
        self.df['fqid'] = self.df['fqid'].astype('category')
        fqid_categories = self.df['fqid'].cat.categories  #
        self.df['fqid'] = self.df['fqid'].cat.codes
        # text_fqid, contain NaN
        self.df['text_fqid'] = self.df['text_fqid'].astype('category')
        text_fqid_categories = self.df['text_fqid'].cat.categories  #
        self.df['text_fqid'] = self.df['text_fqid'].cat.codes
        # text, contain NaN, 595 kinds
        self.df['text'] = self.df['text'].astype('category')
        text_categories = self.df['text'].cat.categories  #
        self.df['text'] = self.df['text'].cat.codes

        self.var = {'event_name': event_name_categories, 'name': name_categories, 'level_group': level_group_categories,
                    'room_fqid': room_fqid_categories, 'page': page_categories, 'fqid': fqid_categories,
                    'text_fqid': text_fqid_categories, 'text': text_categories, 'mode': self.mode}

        # session_id
        self.df['year'] = self.df['session_id'].apply(lambda x: int(str(x)[:2])).astype(np.uint8)
        self.df['month'] = self.df['session_id'].apply(lambda x: int(str(x)[2:4]) + 1).astype(np.uint8)
        self.df['weekday'] = self.df['session_id'].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
        self.df['hour'] = self.df['session_id'].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
        self.df['minute'] = self.df['session_id'].apply(lambda x: int(str(x)[8:10])).astype(np.uint8)
        self.df['second'] = self.df['session_id'].apply(lambda x: int(str(x)[10:12])).astype(np.uint8)

    def _slimming(self):
        df = self.df
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
        return self

    def get_data(self, level_group):
        group = self.df.groupby('level_group').get_group(level_group)
        return group.reset_index(drop=True)

    def get_var(self, write: bool):
        if write:
            params = {'event_name': list(self.var['event_name']), 'name': list(self.var['name']),
                      'level_group': list(self.var['level_group']), 'room_fqid': list(self.var['room_fqid']),
                      'page': list(self.var['page']), 'fqid': list(self.var['fqid']),
                      'text_fqid': list(self.var['text_fqid']), 'text': list(self.var['text']),
                      'mode': self.var['mode']}
            with open('./data/params.json', 'w') as f:
                json.dump(params, f)
        return self.var

    def drop_loc(self, min_len):
        groups = self.df.groupby(['session_id', 'level_group'])
        deleted_groups = groups.filter(lambda x: len(x) < min_len)
        self.df = groups.filter(lambda x: len(x) >= min_len)
        return deleted_groups


def test_processing(dataframe, vars_dict, seq_len):
    columns1 = ['index', 'index_time', 'room_coor_x', 'room_coor_y', 'screen_coor_x',
                'screen_coor_y', 'hover_duration', 'event_name', 'name', 'level', 'page', 'fqid', 'room_fqid',
                'text_fqid', 'level_group', 'text']
    columns2 = ['year', 'month', 'weekday', 'hour', 'minute', 'second']
    df = dataframe

    def _quantize():
        var = vars_dict
        # index
        df['index'] = df.groupby('session_id').cumcount() + 1

        # elapsed_time, convert ms to s by divide by 1000, drop the index whose elapsed_time > 6000
        df['elapsed_time'] = df['elapsed_time'].apply(lambda x: x / 1000).astype(np.float32)
        df.drop(df[df['elapsed_time'] > 6000].index, inplace=True)
        df['index_time'] = df.groupby('session_id')['elapsed_time'].diff().fillna(
            df['elapsed_time']).astype(np.float32)

        # fill nan with median for numerical columns
        cols_to_fill = ['room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']
        df[cols_to_fill] = df[cols_to_fill].fillna(var['mode'])

        # category map
        # event_name, no NaN
        df['event_name'] = pd.Categorical(df['event_name'], var['event_name'])
        df['event_name'] = df['event_name'].cat.codes
        # name, 6 category, no NaN
        df['name'] = pd.Categorical(df['name'], var['name'])
        df['name'] = df['name'].cat.codes
        # level, numeric
        # level_group, 3 category, no NaN
        df['level_group'] = pd.Categorical(df.level_group, ['0-4', '5-12', '13-22'])
        df['level_group'] = df['level_group'].cat.codes
        # room_fqid, no NaN
        df['room_fqid'] = pd.Categorical(df['room_fqid'], var['room_fqid'])
        df['room_fqid'] = df['room_fqid'].cat.codes
        # page, contain NaN
        df['page'] = pd.Categorical(df['page'], var['page'])
        df['page'] = df['page'].cat.codes
        # fqid, contain NaN, 128 kinds including NaN
        df['fqid'] = pd.Categorical(df['fqid'], var['fqid'])
        df['fqid'] = df['fqid'].cat.codes
        # text_fqid, contain NaN
        df['text_fqid'] = pd.Categorical(df['text_fqid'], var['text_fqid'])
        df['text_fqid'] = df['text_fqid'].cat.codes
        # text, contain NaN, 595 kinds
        df['text'] = pd.Categorical(df['text'], var['text'])
        df['text'] = df['text'].cat.codes

        # session_id
        df['year'] = df['session_id'].apply(lambda x: int(str(x)[:2])).astype(np.uint8)
        df['month'] = df['session_id'].apply(lambda x: int(str(x)[2:4]) + 1).astype(np.uint8)
        df['weekday'] = df['session_id'].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
        df['hour'] = df['session_id'].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
        df['minute'] = df['session_id'].apply(lambda x: int(str(x)[8:10])).astype(np.uint8)
        df['second'] = df['session_id'].apply(lambda x: int(str(x)[10:12])).astype(np.uint8)

    _quantize()

    if len(df) < seq_len:
        df = df.reindex(index=list(range(seq_len)), fill_value=-1)

    level_group = df['level_group'][0]

    data_x = df[columns1].values
    seq_num = data_x.shape[0]
    begin, end = 0, seq_len
    stamp = df[columns2][begin:end].values

    batch_size = seq_num - seq_len + 1
    batch_size = min(batch_size, 256)
    batch_data = np.zeros((batch_size, seq_len, len(columns1)))
    batch_stamp = np.zeros((batch_size, seq_len, 6))
    for i in range(batch_size):
        batch_data[i, :, :] = data_x[begin:end, :]  # Front closed and rear open
        batch_stamp[i, :, :] = stamp
        begin += 1
        end += 1
    return torch.FloatTensor(batch_data), torch.FloatTensor(batch_stamp), level_group


class MyDataset(Dataset):
    def __init__(self, dataframe, labels, flag, seq_len, columns=None):
        if columns is None:
            columns = ['index', 'index_time', 'room_coor_x', 'room_coor_y', 'screen_coor_x',
                       'screen_coor_y', 'hover_duration', 'event_name', 'name', 'level', 'page', 'fqid', 'room_fqid',
                       'text_fqid', 'level_group', 'text']
        self.columns = columns
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]  # 0 or 1
        self.df = dataframe
        self.labels_df = labels
        self.seq_len = seq_len
        self.questions = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
                          'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18']
        self._read_data()

    def _read_data(self):
        self.labels = self.labels_df.set_index('session_id')['correct'].to_dict()
        session = self.df['session_id'].unique()  # ndarray
        mid = int(len(session)*0.6)
        border1 = [0, mid]
        border2 = [mid, len(session)]
        self.session = session[border1[self.set_type]:border2[self.set_type]]
        self.groups = self.df.groupby('session_id')

    def __getitem__(self, item):
        session_id = self.session[item]
        session_q = [str(session_id) + '_' + q for q in self.questions]
        data_y = [self.labels[sq] for sq in session_q]
        group = self.groups.get_group(session_id).reset_index(drop=True)
        start = np.random.randint(0, len(group) - self.seq_len + 1)
        data = group[start:start + self.seq_len]
        stamp = data[['year', 'month', 'weekday', 'hour', 'minute', 'second']].values
        data = data[self.columns].values
        return torch.FloatTensor(data), torch.FloatTensor(data_y), torch.FloatTensor(stamp)

    def __len__(self):
        return len(self.session)
