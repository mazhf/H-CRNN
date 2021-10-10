# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:03:00 2020

@author: mazhf
"""

import pandas as pd
import joblib
import os

current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, "../.."))
print('current_dir_path: ', current_dir_path)
print('parent_dir_path: ', parent_dir_path)

read_path_csv = os.path.join(parent_dir_path, 'clean_data', 'cut_data.csv')
save_parent_path = os.path.join(parent_dir_path, 'clean_data', 'split_preprocess_data')

df = pd.read_csv(read_path_csv)
spilt_num = 5
batch = df.shape[0] // spilt_num
for i in range(spilt_num):
    if i == 0:
        df_split = df.loc[0:(i + 1) * batch - 1, 'data']
    if i == (spilt_num - 1):
        df_split = df.loc[i * batch:, 'data']
    else:
        df_split = df.loc[i * batch:(i + 1) * batch - 1, 'data']
    save_path = os.path.join(save_parent_path, 'split_' + str(i) + '.pkl')
    print('df_split' + str(i) + '_shape: ', df_split.shape)
    joblib.dump(df_split, save_path)

'''
爬坑心得：
loc主要用于文字标签定位，故输入为离散型数据，则0:1代表[0,1]，
而iloc主要用于数字标签定位，故输入为连续型数据，则0:1代表[0,1)，
但两者都可索引离散型和连续性数据
'''

