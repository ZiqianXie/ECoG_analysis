# -*- coding: utf-8 -*-
import h5py
import os
from sklearn.linear_model import ElasticNetCV, LogisticRegression
import numpy as np


os.chdir(os.path.dirname(__file__))


def fun(x):
    c = 0
    x2 = np.empty_like(x)
    for i in range(x.shape[0]):
        c *= np.exp(-1/15)
        c += x[i]
        c = min(c, 1)
        x2[i] = c
    return x2

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
                    idxb = f2[sid]["finger"+str(finger+1)][method+"_l"][:]-1
                    en = ElasticNetCV()
                    en.fit(X[:, idxc].astype("float64"), Y[:, finger])
                    lr = LogisticRegression()
                    lr.fit(X[:, idxb], Yb[:, finger])
                    yp = en.predict(Xt[:, idxc])*fun(lr.predict(Xt[:, idxb]))
                    corr = np.corrcoef(yp, Yt[:, finger])[0, 1]
                    with open("log.txt", "a") as log:
                        print >> log, sid, " finger", finger,\
                                " ", method, ":", corr
