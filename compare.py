# -*- coding: utf-8 -*-
import h5py
from sklearn.linear_model import OrthogonalMatchingPursuitCV
import numpy as np
import pickle


omp = OrthogonalMatchingPursuitCV()
f1 = [0, 1, 3, 4]
f2 = [0, 3, 4]


def calcomp(sid, fingers):
    c = []
    idx = []
    with h5py.File("ECoG_big_data.h5", "r") as f:
        X = f["sub{}/train_data".format(sid)][:]
        Y = f["sub{}/train_clabel".format(sid)][:]
        Xt = f["sub{}/test_data".format(sid)][:]
        Yt = f["sub{}/test_clabel".format(sid)][:]
        for f in fingers:
            omp.fit(X, Y[:, f])
            c.append(np.corrcoef(omp.predict(Xt), Yt[:, f])[0, 1])
            idx.append(omp.coef_.nonzero())
    return c, idx

r1 = calcomp(1, f1)
r2 = calcomp(2, f2)
pickle.dump({"r1": r1, "r2": r2}, open("omp_result", "wb"))
