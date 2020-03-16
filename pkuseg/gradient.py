import pkuseg.model
from typing import List

import pkuseg.inference as _inf
import pkuseg.data


def get_grad_SGD_minibatch(grad: List[float], model: pkuseg.model.Model, X: List[pkuseg.data.Example]):
    """
    :param X: 当前最小批的样本Example对象的列表, []
    """
    all_id_set = set()
    errors = 0
    for x in X:
        error, id_set = get_grad_CRF(grad, model, x)
        errors += error
        all_id_set.update(id_set)

    return errors, all_id_set


def get_grad_CRF(grad: List[float], model: pkuseg.model.Model, x: pkuseg.data.Example):
    """
    :param x: 单个样本, 类型是 pkuseg.data.Example
    """
    id_set = set()

    n_tag = model.n_tag  # 5个标签
    bel = _inf.belief(len(x), n_tag)  # len(x) 得到的是每行文本中 特征行的个数, 也就是每行中字符的个数
    belMasked = _inf.belief(len(x), n_tag)

    Ylist, YYlist, maskYlist, maskYYlist = _inf.getYYandY(model, x)
    Z, sum_edge = _inf.get_beliefs(bel, model, x, Ylist, YYlist)
    ZGold, sum_edge_masked = _inf.get_beliefs(belMasked, model, x, maskYlist, maskYYlist)

    for i, node_feature_list in enumerate(x.features):
        for feature_id in node_feature_list:
            trans_id = model._get_node_tag_feature_id(feature_id, 0)
            id_set.update(range(trans_id, trans_id + n_tag))
            grad[trans_id:trans_id + n_tag] += bel.belState[i] - belMasked.belState[i]

    backoff = model.n_feature * n_tag
    grad[backoff:] += sum_edge - sum_edge_masked
    id_set.update(range(backoff, backoff + n_tag * n_tag))

    return Z - ZGold, id_set
