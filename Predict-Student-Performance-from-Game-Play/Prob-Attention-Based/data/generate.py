import pandas as pd
import json


CATS = ['event_name', 'name', 'page', 'fqid', 'room_fqid', 'text_fqid', 'text']
df = pd.read_csv('../../data/train.csv', usecols=CATS)

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


with open('./test.json', 'w') as f:
    json.dump(cat_map, f)


