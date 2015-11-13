# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
from filter_design import filter_design
from sklearn.linear_model import LassoLarsCV, LogisticRegressionCV
from collections import Counter
import cPickle as pickle


def nonzero_seg(x):
    '''find start and end indices of nonzero segments of a 1D array'''
    nzid = x.nonzero()[0]
    tmp = np.where(np.diff(nzid) != 1)[0]
    return np.vstack((np.hstack((nzid[0], nzid[tmp + 1])),
                      np.hstack((nzid[tmp], nzid[-1])))).T


def striding(data, lens, stride):
    from numpy.lib.stride_tricks import as_strided as ast
    itemsize = data.itemsize
    t, chan = data.shape
    n = np.floor((t-lens)/stride) + 1
    if data.flags.c_contiguous:
        strides = tuple(itemsize*i for i in (chan*stride, 1, chan))
    else:
        strides = tuple(itemsize*i for i in (stride, t, 1))
    return ast(data, shape=(n, chan, lens), strides=strides)


def binarize(dg, thres1=30, thres2=10):
    dg = dg.copy()
    if dg.ndim == 1:
        dg = dg.reshape(-1, 1)
    dg[dg > 0] = 1
    for col in dg.T:
        nzseg = nonzero_seg(col)
        end = nzseg[1:, 0]
        start = nzseg[:-1, 1]
        idx = np.where(end - start < thres1)[0]
        for id in idx:
            col[start[id]:end[id]] = 1
        nzseg = nonzero_seg(col)
        start = nzseg[:, 0]
        end = nzseg[:, 1]
        idx = np.where(end - start < thres2)[0]
        for id in idx:
            col[start[id]:end[id]+1] = 0
    return dg


def preprocessing(data, dg, delay=7, **kwargs):
    filters = filter_design()
    filtered = np.vstack(np.convolve(d, f, mode='valid')
                         for d in data.T for f in filters).T
    logpower = np.log((striding(filtered, 40, 40)**2).sum(2))
    X = striding(logpower, delay, 1).reshape(logpower.shape[0]-delay+1, -1)
    dg = dg.reshape(-1, 1)
    y = dg[-X.shape[0]:, :]
    yb = binarize(y)
    yb = label_shift_left(yb)
    return (X, y, yb)


def label_shift_left(dg, left1=4, left2=4):
    dg = dg.copy()
    for col in dg.T:
        nzseg = nonzero_seg(col)
        for row in nzseg:
            start = row[0]
            end = row[1]
            col[start-left1:start] = 1
            col[end-left2:end+1] = 0
    return dg


def num2info(num, delay=7, num_filter=3):
    """
    return tuple (chan, f, t)
    """
    block = delay * num_filter
    chan = num / block
    f = (num % block) / delay
    t = (num % block) % delay
    return (chan, f, t)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    subj = 'sub1'
    finger = 1
    with h5py.File('ECoG_data.h5', 'r+') as f:
        u = f[subj]['unmixing_matrix'][:]
        X = f[subj]['train_data'][:]
        X -= X.mean(0)
        X = X.dot(u)
        Y = f[subj]['cleaned_train_dg'][:]
    X1, y1, _ = preprocessing(X, Y[:, finger])
    ls = LassoLarsCV()
    ls.fit(X1, y1[:, 0])
    pickle.dump(ls, open('linear_model_'+subj+'_'+str(finger), 'wb'))
    channel_count = Counter([num2info(c)[0] for c in ls.coef_.nonzero()[0]])
    X2, _, yb = preprocessing(X[:, list(set(channel_count.keys()))],
                              Y[:, finger])
    ls2 = LogisticRegressionCV()
    ls2.fit(X2, yb[:, 0])
    pickle.dump(ls2, open('logistic_model_'+subj+'_'+str(finger), 'wb'))
    with h5py.File('selected_channel.h5', 'w') as f:
            f.create_dataset('selected_channel',
                             data=list(set(channel_count.keys())))
