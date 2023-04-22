import pandas as pd
import numpy as np
import json
import os
from utils import drop_session


use_cols = ['session_id', 'index', 'elapsed_time', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y',
            'hover_duration', 'event_name', 'name', 'level', 'page', 'fqid', 'room_fqid',
            'text_fqid', 'text', 'level_group']

dtypes = {'index': np.int32, 'elapsed_time': np.int32, 'room_coor_x': np.float32,
          'room_coor_y': np.float32, 'screen_coor_x': np.float32, 'screen_coor_y': np.float32,
          'hover_duration': np.float32, 'level': np.int8}

df = pd.read_csv('./data/train.csv', usecols=use_cols, dtype=dtypes)
print('df memory usage: {} MB'.format(df.memory_usage(index=True, deep=True).sum() / 1024**2))

NUMS = ['room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']
CATS = ['event_name', 'name', 'page', 'fqid', 'room_fqid', 'text_fqid', 'text']

# mode = df[NUMS].mode()

event_name_list = list(df['event_name'].value_counts(sort=True).index)
name_list = list(df['name'].value_counts(sort=True).index)
page_list = list(df['page'].value_counts(sort=True).index)
fqid_list = list(df['fqid'].value_counts(sort=True).index)
room_fqid_list = list(df['room_fqid'].value_counts(sort=True).index)
text_list = list(df['text'].value_counts(sort=True).index)
text_fqid_list = list(df['text_fqid'].value_counts(sort=True).index)

event_name_dict = {val: idx+1 for idx, val in enumerate(event_name_list)}
name_dict = {val: idx+1 for idx, val in enumerate(name_list)}
page_dict = {val: idx+1 for idx, val in enumerate(page_list)}
fqid_dict = {val: idx+1 for idx, val in enumerate(fqid_list)}
room_fqid_dict = {val: idx+1 for idx, val in enumerate(room_fqid_list)}
text_dict = {val: idx+1 for idx, val in enumerate(text_list)}
text_fqid_dict = {val: idx+1 for idx, val in enumerate(text_fqid_list)}

cat_map = {'event_name': event_name_dict, 'name': name_dict, 'page': page_dict,
           'fqid': fqid_dict, 'room_fqid': room_fqid_dict, 'text': text_dict, 'text_fqid': text_fqid_dict}

df['year'] = df['session_id'].apply(lambda x: int(str(x)[:2]))
df['month'] = df['session_id'].apply(lambda x: int(str(x)[2:4])+1)
df['weekday'] = df['session_id'].apply(lambda x: int(str(x)[4:6]))
df['hour'] = df['session_id'].apply(lambda x: int(str(x)[6:8]))
for cat in cat_map:
    df[cat] = df[cat].map(cat_map[cat])
df = df.fillna(0)

with open('./data/cat_map.json', 'w') as f:
    json.dump(cat_map, f)

grouped = df.groupby('level_group')
for level in ['0-4', '5-12', '13-22']:
    df_level = grouped.get_group(level)
    df_level, _ = drop_session(df_level, min_len=64)
    df_level.to_csv(f'./data/train_{level}.csv', index=False)

