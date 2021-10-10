# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 11:16:47 2020

@author: mazhf
"""
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import os
from parameter import cfg

# 转换标签为数字
current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, "../.."))
print('current_dir_path: ', current_dir_path)
print('parent_dir_path: ', parent_dir_path)

read_label_path_pkl = os.path.join(parent_dir_path, 'raw_data', 'label_list.pickle')
save_label_path_pkl = os.path.join(parent_dir_path, 'clean_data', 'label_list.pkl')

with open(read_label_path_pkl, 'rb') as f:
    label = pickle.load(f)

map_dic_fine = {"教育": 0, "社会": 1, "时政": 2, "财经": 3, "股票": 4, "房产": 5,
                "家居": 6, "游戏": 7, "科技": 8, "体育": 9, "彩票": 10, "娱乐": 11,
                "时尚": 12, "星座": 13}

label_list = [map_dic_fine[item] for item in label]
joblib.dump(label_list, save_label_path_pkl)
print('1 Done!')

# *************************************** split run ********************************************************************

# 把词按词频降序转换为索引值的形式，词和索引的对应存储在word_index中，不足文本最大词个数的文本，在前面补齐0
read_clean_all_path = os.path.join(parent_dir_path, 'clean_data', 'split_preprocess_data', 'clean_all.pkl')
save_word_index_path = os.path.join(parent_dir_path, 'clean_data', 'word_index.pkl')
save_word_sequences_array_path = os.path.join(parent_dir_path, 'clean_data', 'word_sequences_array.npy')

data = list(joblib.load(read_clean_all_path))
max_reserved_vocab = cfg.max_reserved_vocab

tokenizer = Tokenizer(num_words=max_reserved_vocab, split=',')
tokenizer.fit_on_texts(data)
data = tokenizer.texts_to_sequences(data)
data = pad_sequences(data, maxlen=max_reserved_vocab + 1)  # all_sample_num * max_reserved_vocab+1 为了节省内存
word_index = tokenizer.word_index
# 取前max_reserved_vocab个word_index
word_index = {k: v for k, v in word_index.items() if v < max_reserved_vocab + 1}
np.save(save_word_sequences_array_path, data)
joblib.dump(word_index, save_word_index_path)
print('2 Done!')

# *************************************** split run ********************************************************************

# 划分训练、测试集，训练和测试集中每个类别的比例相等-stratify处理不平衡数据
read_word_sequences_array_path = os.path.join(parent_dir_path, 'clean_data', 'word_sequences_array.npy')
read_label_path_pkl = os.path.join(parent_dir_path, 'clean_data', 'label_list.pkl')
save_train_test_dataset_path = os.path.join(parent_dir_path, 'clean_data', 'train_test_dataset.pkl')
data = np.load(read_word_sequences_array_path)
label = joblib.load(read_label_path_pkl)
x_train, x_test, y_train_fine, y_test_fine = train_test_split(data, label, stratify=label, test_size=0.2,
                                                              random_state=1, shuffle=True)

map_dic_course = {0: 0, 1: 0, 2: 0,
                  3: 1, 4: 1, 5: 1,
                  7: 2, 8: 2, 6: 2, 12: 2, 13: 2,
                  9: 3, 10: 3, 11: 3}
y_train_course = [map_dic_course[item] for item in y_train_fine]
y_test_course = [map_dic_course[item] for item in y_test_fine]

lb = preprocessing.LabelBinarizer()
y_train_fine = lb.fit_transform(y_train_fine)
y_test_fine = lb.fit_transform(y_test_fine)
y_train_course = lb.fit_transform(y_train_course)
y_test_course = lb.fit_transform(y_test_course)

joblib.dump([x_train, x_test, y_train_fine, y_test_fine, y_train_course, y_test_course], save_train_test_dataset_path)
print('3 Done!')

# *************************************** split run ********************************************************************

'''
生成数据集词索引排序对应的词向量矩阵——数据集shape：batch_size×max_reserved_vocab(keras要求的输入，
keras内部处理应该为独热码向量形式batch_size×max_reserved_vocab×max_reserved_vocab);词向量矩阵shape：max_reserved_vocab×embedding_dim;
这样，经过embedding层后的输出为batch_size×max_reserved_vocab×embedding_dim，达到降维的目的
'''
# 词矩阵-第一列为词-按词频降序排列，词的后面为对应向量（字符型）
read_word2vec_embedding_matrix_path = os.path.join(parent_dir_path, 'clean_data', 'word2vec_embedding_matrix.npy')
read_word_index_path = os.path.join(parent_dir_path, 'clean_data', 'word_index.pkl')
save_embedding_matrix_path = os.path.join(parent_dir_path, 'clean_data', 'embedding_matrix.pkl')

word_index = joblib.load(read_word_index_path)
matrix = np.load(read_word2vec_embedding_matrix_path)  # all_vocab_num × embedding_dim
embedding_dim = cfg.embedding_dim
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))  # max_reserved_vocab × embedding_dim
count = 0
for word, i in word_index.items():
    embedding_vector = matrix[matrix[:, 0] == word][:, 1:].astype(np.float64)
    # words not found in embedding index will be all-zeros.
    if embedding_vector.shape != (0, embedding_dim):
        embedding_matrix[i] = embedding_vector
    else:
        count += 1
        print('not found:', word, i)
print('%d 个单词未找到' % count)

# 保存词向量矩阵
joblib.dump(embedding_matrix, save_embedding_matrix_path)
print('4 Done!')
print('All Done!')
