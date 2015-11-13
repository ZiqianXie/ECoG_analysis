# -*- coding: utf-8 -*-
import theano.tensor as T
import lasagne
import lasagne.layers as layers
import h5py
import glob.glob as glob
from ECoG_model import preprocessing


if not glob('logpower.h5'):
    with h5py.File('ECoG_data.h5', 'r+') as f:
        with h5py.File('logpower.h5', 'r+') as g:
            for i in range(1, 4):
                subj = 'sub' + str(i)
                g.create_group(subj)
                u = f[subj]['unmixing_matrix'][:]
                X = f[subj]['train_data'][:]
                X -= X.mean(0)
                X = X.dot(u)
                Y = f[subj]['cleaned_train_dg'][:]
                X, y, _ = 