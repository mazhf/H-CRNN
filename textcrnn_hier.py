# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:55:51 2020

@author: mazhf
"""
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from keras import backend as K
import joblib
from keras import Input
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding, LSTM, Bidirectional, Dropout, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, LearningRateScheduler
import os
from parameter import cfg
from keras.utils import multi_gpu_model
import time
import numpy as np
import tensorflow as tf
from util.utils import step_decay


tf.compat.v1.disable_eager_execution()


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
save_tensorboard_log_path = os.path.join(parent_dir_path, 'log', 'tensorboard_hier')


# load data
x_train, x_test, y_train_fine, y_test_fine, y_train_course, y_test_course = joblib.load(read_train_test_dataset_path)
embedding_matrix = joblib.load(read_embedding_matrix_path)

# hyper-parameters
max_reserved_vocab = cfg.max_reserved_vocab
embedding_dim = cfg.embedding_dim
classes_fine = cfg.classes_fine
classes_course = cfg.classes_course

# init alpha beta
alpha = K.variable(value=cfg.weight[0][0], dtype="float32", name="alpha")
beta = K.variable(value=cfg.weight[0][1], dtype="float32", name="beta")


def cut(x, index):
    return x[:, index:, :]


# model
def textcrnn_hier(max_sequence_length=max_reserved_vocab + 1, embedding_dim=embedding_dim,
                  embedding_matrix=embedding_matrix, num_classes_fine=classes_fine, num_classes_course=classes_course):
    """
    embedding层将每个文本中词替换为词嵌入矩阵的对应词的行向量表示—降维
    """
    inputs = Input(shape=(max_sequence_length,))
    if embedding_matrix is None:
        embedding = Embedding(input_dim=max_sequence_length, output_dim=embedding_dim, trainable=False)(inputs)
    else:
        embedding = Embedding(input_dim=max_sequence_length, output_dim=embedding_dim, weights=[embedding_matrix],
                              trainable=False)(inputs)
    pool_output = []
    pool_output_ = []
    kernel_sizes = cfg.textcnn_filter_size
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=cfg.textcnn_filter_nums, kernel_size=kernel_size, strides=1, activation='relu')(embedding)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
        pool_output_.append(c)
    pool_output_[0] = Lambda(cut, arguments={'index': 2})(pool_output_[0])
    pool_output_[1] = Lambda(cut, arguments={'index': 1})(pool_output_[1])
    x = concatenate([p for p in pool_output])
    x_ = concatenate([p_ for p_ in pool_output_])
    f_course = Flatten()(x)
    f_course = Dropout(0.5)(f_course)
    outputs_course = Dense(num_classes_course, activation='softmax')(f_course)
    bi = Bidirectional(LSTM(units=cfg.rnn_filter_nums))(x_)
    f_fine = Dropout(0.5)(bi)
    outputs_fine = Dense(num_classes_fine, activation='softmax')(f_fine)
    model = Model(inputs=inputs, outputs=[outputs_course, outputs_fine])
    model.summary()
    return model


# change alpha beta
class LossWeightsModifier(Callback):
    def __init__(self, alp, bet):
        self.alpha = alp
        self.beta = bet

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 10:
            K.set_value(self.alpha, cfg.weight[1][0])
            K.set_value(self.beta, cfg.weight[1][1])
        elif epoch == 20:
            K.set_value(self.alpha, cfg.weight[2][0])
            K.set_value(self.beta, cfg.weight[2][1])


reduce_lr = LearningRateScheduler(step_decay)
change_lw = LossWeightsModifier(alpha, beta)
tnsrbrd = TensorBoard(log_dir=save_tensorboard_log_path)

model = textcrnn_hier()
parallel_model = multi_gpu_model(model, gpus=gpu_num, cpu_relocation=cfg.cpu_relocation)
parallel_model.compile(loss="categorical_crossentropy", optimizer=cfg.optimizer, metrics=cfg.metrics)
# parallel_model.compile(loss=[cfg.loss_course, cfg.loss_fine], optimizer=cfg.optimizer, metrics=cfg.metrics)
parallel_model.fit(x_train, [y_train_course, y_train_fine], batch_size=cfg.batch_size, epochs=cfg.epoch,
                   validation_data=(x_test, [y_test_course, y_test_fine]), verbose=cfg.verbose,
                   callbacks=[reduce_lr, change_lw, tnsrbrd])

# test
_, y_test_fine_pred = parallel_model.predict(x_test, batch_size=int(cfg.batch_size), verbose=cfg.verbose)
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