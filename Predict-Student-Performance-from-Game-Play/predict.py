import sys
from enum import Enum
import json
import numpy as np
import pandas as pd
import torch
from net import HFAttention
from utils import test_processing


if sys.platform == 'win32':
    _input_dir = 'data'
    _output_dir = '.'
else:
    _input_dir = '/kaggle/input/predict-student-performance-from-game-play'
    _output_dir = '/kaggle/working'


class State(Enum):
    INIT = 1
    AWAITING_PREDICT = 2
    MADE_PREDICTION = 3
    DONE = 4


def make_env() -> "Competition":
    if make_env.__called__:
        raise Exception("You can only call `make_env()` once.")

    make_env.__called__ = True
    return Competition(pd.read_csv(f'{_input_dir}/test.csv'))


make_env.__called__ = False


class Competition:
    _state: State = State.INIT

    groups = {
        '0-4': list(range(1, 4)),
        '5-12': list(range(4, 14)),
        '13-22': list(range(14, 19)),
    }

    def __init__(self, df):
        df['level_group'] = pd.Categorical(df.level_group, list(self.groups))

        self.df = df

        df_groupby = self.df.sort_values(['level_group', 'session_id']).groupby(['level_group', 'session_id'])
        self.df_iter = df_groupby.__iter__()

        self.predictions = None

    def __iter__(self):
        return self

    def iter_test(self):
        return self

    def __next__(self):
        assert self._state in [State.INIT,
                               State.MADE_PREDICTION], "You must call `predict()` before you get the next batch of data."

        try:
            (level_group, session_id), df = next(self.df_iter)
        except StopIteration:
            self._state = State.DONE
            self.predictions.to_csv('local_submission.csv', index=False)
            raise

        pred_df = pd.DataFrame({
            'session_id': [f'{session_id}_q{q}' for q in self.groups[level_group]],
            'correct': [0 for _ in self.groups[level_group]],
        })

        self._state = State.AWAITING_PREDICT
        return pred_df, df.drop(columns=['session_level']).reset_index(drop=True)

    def predict(self, pred_df):
        assert self._state == State.AWAITING_PREDICT, "You must get the next batch before making a new prediction."
        assert pred_df.columns.to_list() == ['session_id', 'correct'], "Prediction dataframe have invalid columns."

        if self.predictions is not None:
            self.predictions = pd.concat([self.predictions, pred_df])
        else:
            self.predictions = pred_df.copy()

        self._state = State.MADE_PREDICTION


# from data.jo_wilder import make_env
make_env.__called__ = False  # 1: be able to call make_env() again

competition = make_env()
competition._state = competition._state.__class__['INIT']  # 2: Be able to start the competition iteration again

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = HFAttention(seq_len=96, token_size=16, d_model=512)
net.load_state_dict(torch.load("./checkpoint/weights.pth"))
net.to(device)
with open('./data/params.json', 'r') as f:
    vars_dict = json.load(f)

for labels, train in competition.iter_test():
    # run blocks ... do something...
    batch_data, batch_stamp, level_group = test_processing(train, vars_dict, 96)
    predict = net(batch_data.to(device), batch_stamp.to(device))[level_group]
    bin_predict = np.where(predict.cpu() >= 0.5, 1, -1)
    result = np.where(np.sum(bin_predict, axis=0) >= 0, 1, 0)
    labels['correct'] = result
    # predict the results
    competition.predict(labels)
