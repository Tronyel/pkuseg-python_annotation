import random

import numpy as np
import pkuseg.gradient as _grad


# from pkuseg.config import config


class Optimizer:
    def __init__(self):
        self._preVals = []

    def converge_test(self, err):
        val = 1e100
        if len(self._preVals) > 1:
            prevVal = self._preVals[0]
            if len(self._preVals) == 10:
                self._preVals.pop(0)
            avgImprovement = (prevVal - err) / len(self._preVals)
            relAvg = avgImprovement / abs(err)
            val = relAvg
        self._preVals.append(err)
        return val

    def optimize(self):
        raise NotImplementedError()


class ADF(Optimizer):
    def __init__(self, config, dataset, model):
        super().__init__()
        self.config = config
        self._model = model
        self._X = dataset
        """
        self.rate0 = 0.05  # init value of decay rate in SGD and ADF training
        初始化一个与权重矩阵大小一样的矩阵，作为权重衰减矩阵
        """
        self.decayList = np.ones_like(self._model.w) * config.rate0

    def optimize(self):
        config = self.config
        sample_size = 0
        w = self._model.w
        fsize = w.shape[0]
        xsize = len(self._X)  # 得到的是文本中的行数，即句子的个数
        grad = np.zeros(fsize)
        error = 0

        feature_count_list = np.zeros(fsize)
        ri = list(range(xsize))
        random.shuffle(ri)

        """
        self.nUpdate = 10  # for ADF training, 一共更新10次
        """
        update_interval = xsize // config.nUpdate  # 只保留整数部分

        n_sample = 0
        for t in range(0, xsize, config.miniBatch):  # self.miniBatch = 1， 对于每个最小批而言
            XX = []  # 当前最小批使用的所有样本id
            end = False
            for k in range(t, t + config.miniBatch):  # 对于最小批中的每个样本k而言
                i = ri[k]  # 列表ri中的元素已经是打乱后的id
                x = self._X[i]
                XX.append(x)
                if k == xsize - 1:
                    end = True
                    break  # 防止最后一个最小批的样本个数 < 最小批的大小， 而出错
            mb_size = len(XX)
            n_sample += mb_size  # 当前已经选择的累计样本个数

            err, feature_set = _grad.get_grad_SGD_minibatch(
                grad, self._model, XX
            )
            error += err

            feature_set = list(feature_set)

            feature_count_list[feature_set] += 1

            # for i in feature_set:
            #     feature_count_list[i] += 1
            check = False

            for k in range(t, t + config.miniBatch):
                if t != 0 and k % update_interval == 0:
                    check = True

            # update decay rates
            if check or end:
                self.decayList *= (
                        config.upper
                        - (config.upper - config.lower)
                        * feature_count_list
                        / n_sample
                )
                feature_count_list.fill(0)

                # for i in range(fsize):
                #     v = feature_count_list[i]
                #     u = v / n_sample
                #     eta = config.upper - (config.upper - config.lower) * u
                #     self.decayList[i] *= eta
                # feature_count_list
                # for i in range(len(feature_count_list)):
                #     feature_count_list[i] = 0
            # update weights

            w[feature_set] -= self.decayList[feature_set] * grad[feature_set]
            grad[feature_set] = 0
            # for i in feature_set:
            #     w[i] -= self.decayList[i] * grad[i]
            #     grad[i] = 0
            # reg
            if check or end:
                if config.reg != 0:
                    w -= self.decayList * (
                            w / (config.reg * config.reg) * n_sample / xsize
                    )

                    # for i in range(fsize):
                    #     grad_i = (
                    #         w[i] / (config.reg * config.reg) * (n_sample / xsize)
                    #     )
                    #     w[i] -= self.decayList[i] * grad_i
                n_sample = 0
            sample_size += mb_size
        if config.reg != 0:
            s = (w * w).sum()
            error += s / (2.0 * config.reg * config.reg)
        diff = self.converge_test(error)
        return error, sample_size, diff
