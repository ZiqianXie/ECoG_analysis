import os
import h5py
import numpy as np
from clean_label import nonzero_seg
from filter_design import filter_design
from sklearn.linear_model import LassoCV
from utils import num2info
from collections import Counter

os.chdir(os.path.dirname(__file__))


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


def preprocessing(data, dg, delay=7):
    filters = filter_design()
    filtered = np.vstack(np.convolve(d, f, mode='valid')
                         for d in data.T for f in filters).T
    logpower = np.log((striding(filtered, 40, 40)**2).mean(2))
    X = striding(logpower, delay, 1).reshape(logpower.shape[0]-delay+1, -1)
    dg = dg.reshape(-1, 1)
    Y = dg[-X.shape[0]:, :]
    Yb = binarize(Y)
    Yb = label_shift_left(Yb)
    return (X, Y, Yb)


def label_shift_left(dg, left1=4, left2=0):
    dg = dg.copy()
    for col in dg.T:
        nzseg = nonzero_seg(col)
        for row in nzseg:
            start = row[0]
            end = row[1]
            col[start-left1:start] = 1
            col[end-left2:end+1] = 0
    return dg
if __name__ == "__main__":
    filters = filter_design()
    subj = 'sub1'
    finger = 1
    f = h5py.File('ECoG_data.h5', 'r+')
    u = f[subj]['unmixing_matrix'][:]
    X = f[subj]['train_data'][:].dot(u)
    Y = f[subj]['cleaned_train_dg'][:]
    Xt = f[subj]['test_data'][:].dot(u)
    Yt = f[subj]['cleaned_test_dg'][:]
    f.close()
    X, y, _ = preprocessing(X, Y[:, finger])
    Xt, yt, _ = preprocessing(Xt, Yt[:, finger])
    ls = LassoCV()
    ls.fit(X, y[:, 0])
    print ls.score(Xt, yt[:, 0])
    channel_count = Counter([num2info(c)[0] for c in ls.coef_.nonzero()[0]])
    print [c[0] for c in channel_count.items()]
    print channel_count.most_common(10)
