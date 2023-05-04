import json
import sys
from enum import Enum
import numpy as np
import pandas as pd
import torch
from net import HFAttention
from utils import preprocess


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
iter_test = competition.iter_test()
competition._state = competition._state.__class__['INIT']  # 2: Be able to start the competition iteration again
with open('./data/cat_map.json') as f:
    cat_map = json.load(f)
device = torch.device('cpu')
net = HFAttention(seq_len=96, token_size=19, d_model=512)
net.load_state_dict(torch.load("./checkpoint/weights.pth"))
net.to(device)
net_rank = {'0-4': 0, '5-12': 1, '13-22': 2}
question_rank = {'0-4': (1, 4), '5-12': (4, 14), '13-22': (14, 19)}
for sample_submission, train in iter_test:
    # run model ... do something...
    level_group = train.level_group.values[0]
    data_x = preprocess(train, cat_map, seq_len=96)
    net.eval()
    with torch.no_grad():
        y_pred = net(data_x)[net_rank[level_group]]
        y_pred = torch.sigmoid(y_pred)
        y_pred_logit = np.atleast_2d(np.where(y_pred >= 0.625, 1, 0))

    sample_submission['question'] = [int(label.split('_')[1][1:]) for label in sample_submission['session_id']]
    a, b = question_rank[level_group]
    for t in range(a, b):
        mask = sample_submission.question == t
        sample_submission.loc[mask, 'correct'] = y_pred_logit[:, t - a]
    # predict the results
    competition.predict(sample_submission[['session_id', 'correct']])

df = pd.read_csv('local_submission.csv')
print(df, '\n', df.dtypes)
