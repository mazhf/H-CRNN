# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 22:57:47 2020

@author: mazhf
"""

import pandas as pd
import joblib
import os

current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, "../.."))
print('current_dir_path: ', current_dir_path)
print('parent_dir_path: ', parent_dir_path)
read_parent_path = os.path.abspath = (parent_dir_path, 'clean_data', 'split_preprocess_data')
save_pkl_path = os.path.abspath = (parent_dir_path, 'clean_data', 'split_preprocess_data', 'clean_all.pkl')

spilt_num = 5
df_all = pd.Series([])
df_all.rename("data", inplace=True)

for i in range(spilt_num):
    read_file_name = 'split_clean_' + str(i) + '.pkl'
    read_path = os.path.join(read_parent_path, read_file_name)
    df_split = joblib.load(read_path)
    df_all = df_all.append(df_split)

joblib.dump(df_all, save_pkl_path)

# df = joblib.load(save_path)
