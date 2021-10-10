# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:16:34 2020

@author: mazhf
"""

import pandas as pd
import os

current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, "../.."))
print('current_dir_path: ', current_dir_path)
print('parent_dir_path: ', parent_dir_path)

read_path_csv = os.path.join(parent_dir_path, 'clean_data', 'cut.csv')
save_data_path_csv = os.path.join(parent_dir_path, 'clean_data', 'cut_data.csv')
save_label_path_csv = os.path.join(parent_dir_path, 'clean_data', 'cut_label.csv')

df = pd.read_csv(read_path_csv)

df_label = df[['label_course', 'label_fine']]
df_data = df[['data']]
df_label.to_csv(save_label_path_csv, index=False)
df_data.to_csv(save_data_path_csv, index=False)
