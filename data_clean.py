# -*- coding: utf-8 -*-
import numpy as np
from operator import add


def clean(X, thres=15, length=400, stride=40):
    """
    X is of shape (time/samples, channels)
    it calculates the square difference of the
    original signal and reject the top 0.05 percentile.
    return the row number of the remaining data matrix.
    """
    dX2 = (X[1:, :]-X[:-1, :])**2
    tmp = dX2 > np.percentile(dX2, 99.95, 0)
    idx = np.unique((np.where(np.sum(tmp, 1) > thres)[0] + 1) / stride)

    def r(x):
        return range(max(0, x - length/stride + 1), x)
    return [i for i in range((X.shape[0] - length)/stride + 1) if i not in
            np.unique(reduce(add, map(r, idx), []))]
