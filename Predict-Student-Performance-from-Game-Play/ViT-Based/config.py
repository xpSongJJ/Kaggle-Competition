import json
BATH_SIZE = 32
SEQ_LEN = 384
TOKEN_SIZE = 10
EPOCH = 16
lr = 5e-2
use_cols = ['elapsed_time', 'elapsed_time_diff', 'event_name', 'name', 'level', 'page', 'fqid', 'room_fqid',
            'text', 'text_fqid']
read_cols = ['session_id', 'elapsed_time', 'event_name', 'name', 'level', 'page', 'fqid', 'room_fqid',
             'text', 'text_fqid', 'level_group']

LEVEL_GROUP = ['0-4', '5-12', '13-22']
LEVEL_GROUP_QST = {"0-4": list(range(1, 4)), "5-12": list(range(4, 14)), "13-22": list(range(14, 19))}

with open('./data/cat2codes.json', 'r') as f:
    cat2codes = json.load(f)
