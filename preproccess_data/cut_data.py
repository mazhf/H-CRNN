# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:26:38 2020

@author: mazhf
"""
import pandas as pd
import jieba
import os


current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, "../.."))
print('current_dir_path: ', current_dir_path)
print('parent_dir_path: ', parent_dir_path)

read_path_excel = os.path.join(parent_dir_path, 'raw_data', 'all.xlsx')
save_path_csv = os.path.join(parent_dir_path, 'clean_data', 'cut.csv')

# 读取数据
df = pd.read_excel(read_path_excel).astype(str)

# jieba分词，去除词首位空白
df['data'] = df['data'].map(lambda x: jieba.lcut(x)).map(lambda x: list(map(str.strip, x)))
df.rename(columns={"label": "label_fine"}, inplace=True)

# 打上粗、细类别标签
map_dic_fine = {"教育": 0, "社会": 1, "时政": 2, "财经": 3, "股票": 4, "房产": 5,
                "家居": 6, "游戏": 7, "科技": 8, "体育": 9, "彩票": 10, "娱乐": 11,
                "时尚": 12, "星座": 13}

map_dic_course = {"教育": 0, "社会": 0, "时政": 0,
                  "财经": 1, "股票": 1, "房产": 1,
                  "家居": 2, "游戏": 2, "科技": 2, "时尚": 2, "星座": 2,
                  "体育": 3, "彩票": 3, "娱乐": 3}

df['label_course'] = df['label_fine'].map(map_dic_course)
df['label_fine'] = df['label_fine'].map(map_dic_fine)
df.to_csv(save_path_csv, index=False)
