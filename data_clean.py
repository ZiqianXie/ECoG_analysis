# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os
from ECoG_model import preprocessing
from sklearn.linear_model import LassoLarsCV


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
            np.unique(reduce(lambda x, y: x+y, map(r, idx)))]
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subj = 'sub1'
    finger = 0
    with h5py.File('ECoG_data.h5', 'r+') as f:
        Xr = f[subj]['train_data'][:]
        Y = f[subj]['cleaned_train_dg'][:]
        Xtr = f[subj]['test_data'][:]
        Yt = f[subj]['cleaned_test_dg'][:]
    cidx = clean(Xr)
    meanXr = np.mean(Xr, axis=0)
    stdXr = np.std(Xr, axis=0)
    Xr = (Xr-meanXr)/stdXr
    Xtr = (Xtr-meanXr)/stdXr
    X, y, _ = preprocessing(Xr, Y[:, finger])
    X, y = X[cidx, :], y[cidx, :]
    Xt, yt, _ = preprocessing(Xtr, Yt[:, finger])
    ls = LassoLarsCV()
    ls.fit(X, y[:, 0])
    channel = ls.coef_.nonzero()[0]
