# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:00:37 2020

@author: mazhf
"""

import joblib
from gensim.models import Word2Vec
import numpy as np
from parameter import cfg
import os


current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, "../.."))
print('current_dir_path: ', current_dir_path)
print('parent_dir_path: ', parent_dir_path)

read_pkl_path = os.path.join(parent_dir_path, 'clean_data', 'split_preprocess_data', 'clean_all.pkl')
save_npy_path = os.path.join(parent_dir_path, 'clean_data', 'word2vec_embedding_matrix.npy')

df = joblib.load(read_pkl_path)

embedding_dim = cfg.embedding_dim
# embedding_dim = 150
batch_size = 1024
model = Word2Vec(sentences=df, max_vocab_size=None, vector_size=embedding_dim, window=5, min_count=5,
                 sg=1, hs=1, sample=0.001, negative=0, epochs=30, batch_words=batch_size, alpha=0.025, min_alpha=0.0001,
                 workers=24, sorted_vocab=1)

try:
    model.save(os.path.join(parent_dir_path, 'clean_data', 'word2vec.model'))
except Exception as e:
    print(e)

# model = Word2Vec.load("word2vec.model")

word = np.array(model.wv.index_to_key)[:, np.newaxis]
embedding = model.wv.vectors
matrix = np.concatenate([word, embedding], axis=1)
print(matrix.shape)
np.save(save_npy_path, matrix)
