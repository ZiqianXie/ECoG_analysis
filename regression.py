# -*- coding: utf-8 -*-
import h5py
import os
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
import numpy as np
from scipy.signal import butter, filtfilt


os.chdir(os.path.dirname(__file__))


def fun(x):
    c = 0
    x2 = np.empty_like(x, dtype="f")
    for i in range(x.shape[0]):
        c *= np.exp(-1/30.)
        c += x[i]
        c = min(c, 1)
        x2[i] = c
    return x2


def relu(x):
    return np.where(x < 0, 0, x)
l = []
with h5py.File("ECoG_big_data.h5", "r+") as f1:
    with h5py.File("selected.h5", "r+") as f2:
        for i in range(1, 4):
            sid = "sub"+str(i)
            X = f1[sid]["train_data"][:]
            Y = f1[sid]["train_clabel"][:]
            Yb = f1[sid]["train_blabel"][:]
            Xt = f1[sid]["test_data"][:]
            Yt = f1[sid]["test_clabel"][:]
            Ytb = f1[sid]["test_blabel"][:]
            for finger in range(5):
                for method in ["l1", "mcp", "scad"]:
                    idxc = f2[sid]["finger"+str(finger+1)][method][:]-1
                    idxb = f2[sid]["finger"+str(finger+1)]["l1_l"][:]-1
                    en = ElasticNetCV()
                    en.fit(X[:, idxc].astype("float64"), Y[:, finger])
                    yp = en.predict(Xt[:, idxc])
                    corr = np.corrcoef(yp, Yt[:, finger])[0, 1]
                    if corr < 0.3:
                        break
                    else:
                        l.append([sid+"//"+"finger"+str(finger+1), corr])
                        lr = LogisticRegressionCV()
                        lr.fit(X[:, idxc], Yb[:, finger])
                        tp = yp*fun(lr.predict(Xt[:, idxc]))
                        m = np.where(np.convolve(tp, np.ones((40,))/40,
                                                 mode='same') < 0.5, 0, 1)
                        b, a = butter(2, 9.0/25, 'low')
                        yy = relu(filtfilt(b, a, tp*m))
                        print corr, \
                            np.corrcoef(Yt[:, finger], yy)[0, 1]
