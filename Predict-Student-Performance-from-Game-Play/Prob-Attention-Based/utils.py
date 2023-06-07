import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


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


def test_processing(dataframe, vars_dict, seq_len):
    columns1 = ['index', 'elapsed_time', 'room_coor_x', 'room_coor_y', 'screen_coor_x',
                'screen_coor_y', 'hover_duration', 'event_name', 'name', 'level', 'page', 'fqid', 'room_fqid',
                'text_fqid', 'level_group', 'text']  # 16 columns
    columns2 = ['year', 'month', 'weekday', 'hour', 'minute', 'second']
    df = dataframe

    def _quantize():
        var = vars_dict
        # index, not increment from 1 by session_id/level_group
        # df['index'] = df.groupby('session_id').cumcount() + 1

        # elapsed_time, convert ms to s by divide by 1000, drop the index whose elapsed_time > 6000
        df['elapsed_time'] = df['elapsed_time'].apply(lambda x: x / 1000).astype(np.float32)
        # df.drop(df[df['elapsed_time'] > 6000].index, inplace=True)
        # df['index_time'] = df.groupby('session_id')['elapsed_time'].diff().fillna(
        #     df['elapsed_time']).astype(np.float32)

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
        # len of filled df is 1 longer than prototype so that can be as a batch
        df = df.reindex(index=list(range(seq_len + 1)), fill_value=-1)

    level_group = df['level_group'][0]

    data_x = df[columns1].values
    seq_num = data_x.shape[0]
    begin, end = 0, seq_len
    stamp = df[columns2][begin:end].values

    batch_size = seq_num - seq_len + 1
    batch_size = min(batch_size, 96)
    batch_data = np.zeros((batch_size, seq_len, len(columns1)))
    batch_stamp = np.zeros((batch_size, seq_len, 6))
    for i in range(batch_size):
        batch_data[i, :, :] = data_x[begin:end, :]  # Front closed and rear open
        batch_stamp[i, :, :] = stamp
        begin += 1
        end += 1
    return torch.FloatTensor(batch_data), torch.FloatTensor(batch_stamp), level_group


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


def prefix_na(cat: list, na):
    if na in cat:
        cat.insert(0, cat.pop(cat.index(na)))


def preprocess(dataframe, cat_map: dict, seq_len: int):
    df = dataframe
    df['year'] = df['session_id'].apply(lambda x: int(str(x)[:2]))
    df['month'] = df['session_id'].apply(lambda x: int(str(x)[2:4]) + 1)
    df['weekday'] = df['session_id'].apply(lambda x: int(str(x)[4:6]))
    df['hour'] = df['session_id'].apply(lambda x: int(str(x)[6:8]))
    for cat in cat_map:
        df[cat] = df[cat].map(cat_map[cat])
    df = df.drop(columns=["level_group", "fullscreen", "hq", "music"])
    df = df.fillna(0)
    session_ids = df['session_id'].unique()
    data = np.zeros((len(session_ids), seq_len, 19))
    for i, session_id in enumerate(session_ids):
        df_filtered = df[df['session_id'] == session_id]
        data_x = df_filtered.drop(columns=['session_id']).values
        if data_x.shape[0] < seq_len:
            pad_len = seq_len - data_x.shape[0]
            data_x = np.pad(data_x, ((0, pad_len), (0, 0)))  # padding 0
        else:
            data_x = data_x[:seq_len, :]
        data[i, :, :] = data_x
    return torch.FloatTensor(data)


def drop_session(dataframe, min_len):
    df = dataframe
    groups = df.groupby(['session_id', 'level_group'])
    deleted_groups = groups.filter(lambda x: len(x) < min_len)
    df = groups.filter(lambda x: len(x) >= min_len)
    return df, deleted_groups


def best_f1_threshold(targets: list, pred: list):
    best_threshold = None
    best_f1 = 0
    for thred in np.arange(0.2, 0.8, 0.02):
        pred_logit = [1 if torch.sigmoid(p) >= thred else 0 for p in pred]
        f1 = f1_score(targets, pred_logit)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thred
    return best_f1, best_threshold
