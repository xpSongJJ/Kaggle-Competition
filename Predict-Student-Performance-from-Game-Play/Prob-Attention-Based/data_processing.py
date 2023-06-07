import pandas as pd
import numpy as np
import os


def reduce_memory_usage(df):
    start_mem = df.memory_usage(index=True, deep=True).sum() / 1024**2
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
    mem_usg = df.memory_usage(index=True, deep=True).sum() / 1024**2
    print(f"Memory usage of became: {round(mem_usg, 2)} MB")
    return df


data_root = 'G:/SXP/data/predict-student-performance-from-game-play'
data_name = 'train.csv'
data_path = os.path.join(data_root, data_name)
seq_len = 96
level_group = 0  # alter 0, 1 or 2 correspond to '0-4', '5-12' or '13-22'

df = pd.read_csv(data_path)
"""
    数据集共有20个列标签，其中标签"full_screen","hq"和"music"列标签，无论是在训练集还是在测试集均为空，均为空，所以可删除。
    数据集详情分析见：https://www.kaggle.com/code/demche/student-performance-from-game-play-eda
"""
df.drop(columns=['fullscreen', 'hq', 'music'], inplace=True)
df = reduce_memory_usage(df)  # reduce memory usage
"""
    剩余17个列标签，进行分类：
    3个有关时间的标签：['session_id', 'index', 'elapsed_time'];
    5个纯数值标签: ['room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']; 均含有缺省值
    6个类型标签: ['event_name', 'name', 'level', 'page', 'fqid', 'room_fqid', 'text_fqid', 'level_group']; 'page', fqid','text_fqid'含有缺省值
    1个内容标签：['text']; 含有缺省值
"""

# 'index'标签信息存在错误，同一个session_id下，index递增且不可重复。该标签自认为可以丢弃
df['index'] = df.groupby('session_id').cumcount() + 1

# 讲'elapsed_time'项缩小1000倍，单位换成秒，删除超过6000秒的行，添加新的列'index_time'，表示每个index花费的时间，随后'elapsed_time'可以选择丢弃
df['elapsed_time'] = df['elapsed_time'].apply(lambda x: x / 1000).astype(np.float64)
df.drop(df[df['elapsed_time'] > 6000].index, inplace=True)
df['index_time'] = df.groupby('session_id')['elapsed_time'].diff().fillna(df['elapsed_time'])
df.drop(columns=['elapsed_time'], inplace=True)

# 对数值型缺省值进行填充，填充中位数
cols_to_fill = ['room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']
median = df[cols_to_fill].median()
df[cols_to_fill] = df[cols_to_fill].fillna(median)

# 将类型标签映射到相应的数字
# event_name， 11类, 无缺省值
df['event_name'] = df['event_name'].cat.codes
# name， 6类， 无缺省值
df['name'] = df['name'].cat.codes
# level 本就是数字
# level_group
df['level_group'] = df['level_group'].astype('object')
level_group_map = {'0-4': 0, '5-12': 1, '13-22': 2}
df['level_group'] = df['level_group'].replace(level_group_map)
# room_fqid, 无缺省值
df['room_fqid'] = df['room_fqid'].cat.codes
# page 含有缺省值
page_map = {0.: 0, 1.: 1, 2.: 2, 3.: 3, 4.: 4, 5.: 5, 6.: 6, np.nan: 7}
df['page'] = df['page'].replace(page_map)
# fqid 含有缺省值, 128类（包含NaN），test数据集不包含新的种类
df['fqid'] = df['fqid'].cat.codes

# text_fqid 含有缺省值, 测试集中没有新的种类, category类型可以直接映射到相应的数字
df['text_fqid'] = df['text_fqid'].cat.codes

# text 有缺省值，595类，测试集不含新的类型
df['text'] = df['text'].cat.codes

# session_id
df['year'] = df['session_id'].apply(lambda x: int(str(x)[:2])).astype(np.uint8)
df['month'] = df['session_id'].apply(lambda x: int(str(x)[2:4]) + 1).astype(np.uint8)
df['weekday'] = df['session_id'].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
df['hour'] = df['session_id'].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
df['minute'] = df['session_id'].apply(lambda x: int(str(x)[8:10])).astype(np.uint8)
df['second'] = df['session_id'].apply(lambda x: int(str(x)[10:12])).astype(np.uint8)

# split by level_group
df = df[df['level_group'] == level_group]

# 假设输入网络的序列长度为seq_len，那么index数小于seq_len的session_id将是无效信息
df = df.reset_index(drop=True)  # 调整索引
dlt_session = []
pre = 0
for idx in range(len(df)):
    cur = df['index'][idx]
    if cur < pre:
        if pre < seq_len:
            dlt_session.append(df['session_id'][idx-1])
        pre = cur
    else:
        pre = cur
df = df.drop(df[df.session_id.isin(dlt_session)].index)   # 删除dlt_session中所有的行
print(df.isna().any())

