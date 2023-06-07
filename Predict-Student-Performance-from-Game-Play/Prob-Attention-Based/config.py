import json

BATCH_SIZE = 64
TOKEN_SIZE = 10
EPOCH = 16
LR = 5E-4

SEQ_LEN = 192
MODEL_NAME = "Prob-Attention"

DATA_ROOT = "../data/"
read_cols = ['session_id', 'elapsed_time', 'event_name', 'name', 'level', 'page', 'fqid', 'room_fqid',
             'text', 'text_fqid', 'level_group']
use_cols = ['elapsed_time', 'elapsed_time_diff', 'event_name', 'name', 'level', 'page', 'fqid', 'room_fqid',
            'text', 'text_fqid']
LEVEL_GROUP = ['0-4', '5-12', '13-22']
LEVEL_GROUP_QST = {'0-4': range(3), '5-12': range(3, 13), '13-22': range(13, 18)}

# model params
params = {

}

with open('./data/cat_map.json', 'r') as f:
    cat2codes = json.load(f)
