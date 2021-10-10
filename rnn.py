# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:31:21 2020

@author: mazhf
"""
import random
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import joblib
from keras import Input
from keras.layers import Dense, Flatten, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, LearningRateScheduler
import os
from parameter import cfg
from keras.utils import multi_gpu_model
import numpy as np
import time
from util.utils import step_decay


def fix_random(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


fix_random(2021)

# GPU
GPU = cfg.GPU
gpu_num = len(GPU.split(','))
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

# path
train_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
current_dir_path = os.path.abspath(os.path.dirname(__file__))
parent_dir_path = os.path.abspath(os.path.join(current_dir_path, ".."))
read_train_test_dataset_path = os.path.join(parent_dir_path, 'clean_data', 'train_test_dataset.pkl')
read_embedding_matrix_path = os.path.join(parent_dir_path, 'clean_data', 'embedding_matrix.pkl')
save_tensorboard_log_path = os.path.join(parent_dir_path, 'log', 'tensorboard')

# load data
x_train, x_test, y_train_fine, y_test_fine, y_train_course, y_test_course = joblib.load(read_train_test_dataset_path)
embedding_matrix = joblib.load(read_embedding_matrix_path)

# hyper-parameters
max_reserved_vocab = cfg.max_reserved_vocab
embedding_dim = cfg.embedding_dim
classes_fine = cfg.classes_fine


# model
def rnn(max_sequence_length=max_reserved_vocab + 1, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix,
        num_classes=classes_fine):
    """
    embedding层将每个文本中词替换为词嵌入矩阵的对应词的行向量表示—降维
    """
    inputs = Input(shape=(max_sequence_length,))
    if embedding_matrix is None:
        embedding = Embedding(input_dim=max_sequence_length, output_dim=embedding_dim, trainable=False)(inputs)
    else:
        embedding = Embedding(input_dim=max_sequence_length, output_dim=embedding_dim, weights=[embedding_matrix],
                              trainable=False)(inputs)
    x = Bidirectional(LSTM(units=cfg.rnn_filter_nums))(embedding)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


reduce_lr = LearningRateScheduler(step_decay)
tensorboard = TensorBoard(log_dir=save_tensorboard_log_path)

model = rnn()
parallel_model = multi_gpu_model(model, gpus=gpu_num, cpu_relocation=cfg.cpu_relocation)
parallel_model.compile(loss=cfg.loss, optimizer=cfg.optimizer, metrics=cfg.metrics)
parallel_model.fit(x=x_train, y=y_train_fine, batch_size=cfg.batch_size, epochs=cfg.epoch, verbose=cfg.verbose,
                   validation_data=(x_test, y_test_fine), callbacks=[reduce_lr, tensorboard])

# test
y_test_fine_pred = parallel_model.predict(x_test, batch_size=int(cfg.batch_size), verbose=cfg.verbose)
y_test_fine_pred = np.argmax(y_test_fine_pred, axis=1)
y_test_fine = np.argmax(y_test_fine, axis=1)


def evaluate_model(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')  # total
    print('test precision_micro: %.5f' % p_mic)
    print('test recall_micro: %.5f' % r_mic)
    print('test F1_micro: %.5f' % f_mic)
    print('***********************')
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')  # each label mean
    print('test precision_macro: %.5f' % p_mac)
    print('test recall_macro: %.5f' % r_mac)
    print('test F1_macro: %.5f' % f_mac)
    print('***********************')
    p_wei, r_wei, f_wei, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')  # each label weighted
    print('test precision_weighted: %.5f' % p_wei)
    print('test recall_weighted: %.5f' % r_wei)
    print('test F1_weighted: %.5f' % f_wei)
    print('***********************')


evaluate_model(y_test_fine, y_test_fine_pred)

print('confusion_matrix:', '/n', confusion_matrix(y_test_fine, y_test_fine_pred))
