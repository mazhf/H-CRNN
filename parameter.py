from util.ordered_easydict import OrderedEasyDict as edict
from util.utils import multi_category_focal_loss1, Recall, Precision, F1
from keras.optimizers import Adam


cfg = edict()
cfg.GPU = '2, 3'
cfg.max_reserved_vocab = 500
cfg.embedding_dim = 150
cfg.epoch = 70
cfg.batch_size = 128 * 2
cfg.weight = [[0.8, 0.2], [0.2, 0.8], [0, 1]]
# cfg.optimizer = AdamHDOptimizer(alpha_0=0.01)
cfg.optimizer = Adam(lr=0.001)
cfg.alpha_course = [[2.5], [1.3], [3.6], [4.8]]
cfg.alpha_fine = [[0.4], [0.3], [0.3], [0.4], [0.1], [0.8], [0.5], [0.8], [0.1], [1.6], [3], [0.2], [1.6], [6]]
cfg.loss_course = multi_category_focal_loss1(cfg.alpha_course)  # 'categorical_crossentropy'
cfg.loss_fine = multi_category_focal_loss1(cfg.alpha_fine)  # 'categorical_crossentropy'
cfg.verbose = 1
cfg.metrics = [Recall, Precision, F1]  # ['acc']
cfg.cpu_relocation = True
cfg.classes_fine = 14
cfg.classes_course = 4
cfg.textcnn_filter_size = [2, 3, 4]  # [3, 4, 5]
cfg.textcnn_filter_nums = 4
cfg.rnn_filter_nums = 16

map_dic_fine = {"教育": 0, "社会": 1, "时政": 2, "财经": 3, "股票": 4, "房产": 5,
                "家居": 6, "游戏": 7, "科技": 8, "体育": 9, "彩票": 10, "娱乐": 11,
                "时尚": 12, "星座": 13}

