cimport cython
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libc.math cimport exp, log


# cpdef getLogYY(vector[vector[int]] feature_temp, int num_tag, int backoff, np.ndarray[double, ndim=1] w, double scalar):
cpdef getLogYY():
    """
    :param feature_temp: example.features, 类型是List[List[int]]，内层list表示每个字符的特征id集合，外层的list表示当前文本行中所有的字符
    :param num_tag: model.n_tag, 为5
    :param backoff: model.n_feature*model.n_tag, 即：语料中所有特征的数量 × 5
    :param w: model.w, 初始化后的权重矩阵
    :param scalar: 1.0
    :return: 
    """
    cdef:
        np.ndarray[double, ndim=2] edge_score = np.ones((5, 5), dtype=np.float64)