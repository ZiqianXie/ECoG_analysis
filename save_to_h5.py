# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
from ECoG_model import preprocessing, label_shift_left, binarize
from data_clean import clean

os.chdir(os.path.dirname(__file__))
with h5py.File("ECoG_data.h5", "r+") as f1:
    with h5py.File("ECoG_big_data.h5", "w") as f2:
        for i in range(1, 4):
            subj = "sub" + str(i)
            u = f1[subj]["unmixing_matrix"]
            X = f1[subj]['train_data'][:]
            clist = clean(X)
            X = X.dot(u)
            Y = f1[subj]['cleaned_train_dg'][:]
            Xt = f1[subj]['test_data'][:].dot(u)
            Yt = f1[subj]['cleaned_test_dg'][:]
            X, _, yb = preprocessing(X, Y[:, 0], cleaning=False)
            meanX = np.mean(X, axis=0)
            stdX = np.std(X, axis=0)
            X -= meanX
            X /= stdX
            yb = yb[clist, :]
            clabel = Y[-X.shape[0]:, :][clist, :]
            blabel = np.empty_like(clabel, 'i')
            blabel[:, 0] = yb.ravel()
            for i in range(1, 5):
                blabel[:, i] = \
                label_shift_left(binarize(
                Y[-X.shape[0]:, i]))[clist, :].ravel()
            Xt, _, ytb = preprocessing(Xt, Yt[:, 0], cleaning=False)
            Xt -= meanX
            Xt /= stdX
            tclabel = Yt[-Xt.shape[0]:, :]
            tblabel = np.empty_like(tclabel, 'i')
            tblabel[:, 0] = ytb.ravel()
            for i in range(1, 5):
                tblabel[:, i] = label_shift_left(
                binarize(Y[-Xt.shape[0]:, i])).ravel()
            grp = f2.create_group(subj)
            grp.create_dataset("train_data", data=X[clist, :], dtype='f')
            grp.create_dataset("train_clabel", data=clabel, dtype='f')
            grp.create_dataset("train_blabel", data=blabel, dtype='i')
            grp.create_dataset("test_data", data=Xt, dtype='f')
            grp.create_dataset("test_clabel", data=tclabel, dtype='f')
            grp.create_dataset("test_blabel", data=tblabel, dtype='i')
