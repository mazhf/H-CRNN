# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:38:39 2020

@author: mazhf
"""

import pandas as pd
import pickle
import os

current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, "../.."))
print('current_dir_path: ', current_dir_path)
print('parent_dir_path: ', parent_dir_path)

read_path = os.path.join(parent_dir_path, 'raw_data')
save_path_pkl = os.path.join(parent_dir_path, 'raw_data', 'all.pkl')
save_path_excel = os.path.join(parent_dir_path, 'raw_data', 'all.xlsx')
par_lis = os.listdir(read_path)
all_data = []
for par_file_name in par_lis:
    par_path = os.path.join(read_path, par_file_name)
    if par_file_name == 'label_list.pickle':
        with open(par_path, 'rb') as f:
            label = pickle.load(f)
    if par_file_name == 'split_pickle':
        chil_lis = os.listdir(par_path)
        for chil_file_name in chil_lis:
            chil_path = os.path.join(read_path, par_file_name, chil_file_name)
            with open(chil_path, 'rb') as f:
                data = pickle.load(f)
                all_data = all_data + data

df = pd.DataFrame()
df['label'] = label
df['data'] = all_data
df.to_excel(save_path_excel)

with open(save_path_pkl, 'wb') as f:
    pickle.dump(df, f)
