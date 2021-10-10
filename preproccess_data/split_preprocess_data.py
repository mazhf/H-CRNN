# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 21:54:01 2020

@author: mazhf
"""
import pandas as pd
import re
import joblib
import os


def clean_inner(x):
    """
    去除列表内部的[]符号——默认为列表标识而不是符号，影响正则去除
    """
    s = 'mzf'.join(x)
    s = s.replace('[', '')
    s = s.replace(']', '')
    l = s.split('mzf')
    return l


def clean_symbol(item):
    """
    分词后的非汉字、字母字符替换为空
    """
    item = re.sub(r'[^\u4E00-\u9FA5a-zA-Z]', '', item)
    return item


def remove_stop_words(x, stop_words):
    """
    去除停用词和空
    """
    for item in stop_words:
        while item in x:
            x.remove(item)  # 此处可全部去除，区别于for循环去除
    return x


def decode(x):
    x = x.decode('utf-8')
    return x


def remove_blank(line):
    line = line.strip()
    return line


def load_stopwords(path):
    """
    加载停用词，增加空
    """
    with open(path, 'rb') as f:
        lines = f.readlines()
    stop_words = list(map(decode, lines))
    stop_words = list(map(remove_blank, stop_words))
    stop_words.append('')
    stop_words.append('_')  # 正则无法去除
    return stop_words


current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, "../.."))
print('current_dir_path: ', current_dir_path)
print('parent_dir_path: ', parent_dir_path)
read_parent_path = os.path.abspath = (parent_dir_path, 'clean_data', 'split_preprocess_data')
read_stop_words_path = os.path.join(parent_dir_path, 'stopwords-master', 'hit_stopwords.txt')

stop_words = load_stopwords(read_stop_words_path)
spilt_num = 5

for i in range(spilt_num):
    read_file_name = 'split_' + str(i) + '.pkl'
    read_path = os.path.join(read_parent_path, read_file_name)
    df = joblib.load(read_path)
    df = df.map(eval)
    df = df.map(clean_inner)
    df = df.map(lambda x: list(map(clean_symbol, x)))
    df = df.map(lambda x: remove_stop_words(x, stop_words))
    save_file_name = 'split_clean_' + str(i) + '.pkl'
    save_path = os.path.join(read_parent_path, save_file_name)
    joblib.dump(df, save_path)
