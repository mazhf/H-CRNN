from __future__ import division
import tensorflow as tf
import keras.backend as K


def step_decay(epoch):
    if epoch <= 40:
        lr = 0.001
    elif 40 < epoch <= 55:
        lr = 0.0002
    else:
        lr = 0.00005
    print("learning rate changed to %f" % lr)
    return lr


def Precision(y_true, y_pred):
    """accuracy"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision


def Recall(y_true, y_pred):
    """recall"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall


def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1


def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    https://www.cnblogs.com/CheeseZH/p/13519206.html
    https://blog.csdn.net/u011583927/article/details/90716942
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    # alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
    # alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)

    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss1_fixed


class AdamHDOptimizer(tf.train.GradientDescentOptimizer):
    """
    https://github.com/zadaianchuk/HypergradientDescent
    """

    def __init__(self, alpha_0, beta=10 ** (-7), name="HGD", mu=0.99, eps=10 ** (-8), type_of_learning_rate="global"):
        super(AdamHDOptimizer, self).__init__(beta, name=name)

        self._mu = mu
        self._alpha_0 = alpha_0
        self._beta = beta
        self._eps = eps
        self._type = type_of_learning_rate

    def minimize(self, loss, global_step):

        # Algo params as constant tensors
        mu = tf.convert_to_tensor(self._mu, dtype=tf.float32)
        alpha_0 = tf.convert_to_tensor(self._alpha_0, dtype=tf.float32)
        beta = tf.convert_to_tensor(self._beta, dtype=tf.float32)
        eps = tf.convert_to_tensor(self._eps, dtype=tf.float32)

        var_list = tf.trainable_variables()

        # create and retrieve slot variables for:
        # direction of previous step
        ds = [self._get_or_make_slot(var,
                                     tf.constant(0.0, tf.float32, var.get_shape()), "direction", "direction")
              for var in var_list]
        # current learning_rate alpha
        if self._type == "global":
            alpha = self._get_or_make_slot(alpha_0, alpha_0, "learning_rate", "learning_rate")
        else:
            alphas = [self._get_or_make_slot(var,
                                             tf.constant(self._alpha_0, tf.float32, var.get_shape()), "learning_rates",
                                             "learning_rates")
                      for var in var_list]
        #  moving average estimation
        ms = [self._get_or_make_slot(var,
                                     tf.constant(0.0, tf.float32, var.get_shape()), "m", "m")
              for var in var_list]
        vs = [self._get_or_make_slot(var,
                                     tf.constant(0.0, tf.float32, var.get_shape()), "v", "v")
              for var in var_list]
        # power of mu for bias-corrected first and second moment estimate
        mu_power = tf.get_variable("mu_power", shape=(), dtype=tf.float32, trainable=False,
                                   initializer=tf.constant_initializer(1.0))

        # update moving averages of first and second moment:
        grads = tf.gradients(loss, var_list)
        grads_squared = [tf.square(g) for g in grads]
        m_updates = [m.assign(mu * m + (1.0 - mu) * g) for m, g in zip(ms, grads)]  # new means
        v_updates = [v.assign(mu * v + (1.0 - mu) * g2) for v, g2 in zip(vs, grads_squared)]
        mu_power_update = [tf.assign(mu_power, tf.multiply(mu_power, mu))]
        # bais correction of the estimates
        with tf.control_dependencies(v_updates + m_updates + mu_power_update):
            ms_hat = [tf.divide(m, tf.constant(1.0) - mu_power) for m in ms]
            vs_hat = [tf.divide(v, tf.constant(1.0) - mu_power) for v in vs]

        # update of learning rate alpha, main difference between ADAM and ADAM-HD
        if self._type == "global":
            hypergrad = sum([tf.reduce_sum(tf.multiply(d, g)) for d, g in zip(ds, grads)])
            alphas_update = [alpha.assign(alpha - beta * hypergrad)]
        else:
            hypergrads = [tf.multiply(d, g) for d, g in zip(ds, grads)]
            alphas_update = [alpha.assign(alpha - beta * hypergrad) for alpha, hypergrad in zip(alphas, hypergrads)]

        # update step directions
        with tf.control_dependencies(
                alphas_update):  # we want to be sure that alphas calculated using previous step directions
            ds_updates = [d.assign(-tf.divide(m, tf.sqrt(v) + self._eps)) for (m, v, d) in zip(ms_hat, vs_hat, ds)]

        # update parameters of the model
        with tf.control_dependencies(ds_updates):
            if self._type == "global":
                dirs = [alpha * d for d in ds]
                alpha_norm = alpha
            else:
                dirs = [alpha * d for d, alpha in zip(ds, alphas)]
                alpha_norm = sum([tf.reduce_mean(alpha ** 2) for alpha in alphas])
            variable_updates = [v.assign_add(d) for v, d in zip(var_list, dirs)]
            global_step.assign_add(1)
            # add summaries  (track alphas changes)
            with tf.name_scope("summaries"):
                with tf.name_scope("per_iteration"):
                    alpha_norm_sum = tf.summary.scalar("alpha", alpha_norm,
                                                       collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
        return tf.group(*variable_updates)
