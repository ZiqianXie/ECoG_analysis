# -*- coding: utf-8 -*-
import h5py
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize


f = h5py.File('ECoG_data.h5', 'r+')
for subj_id in range(1, 4):
    x1 = f['sub{}/train_data'.format(subj_id)][:]
    x2 = f['sub{}/test_data'.format(subj_id)][:]
    x = np.concatenate((x1, x2))
    x = normalize(x)
    fica = FastICA(max_iter=10000)
    fica.fit(x)
    unmixing_matrix = fica.components_.T
    f['sub{}/unmixed_train_data'.format(subj_id)][:] = x1.dot(unmixing_matrix)
    f['sub{}/unmixed_test_data'.format(subj_id)][:] = x2.dot(unmixing_matrix)
    f['sub{}/unmixing_matrix'.format(subj_id)][:] = unmixing_matrix
    print '{}/3 done'.format(subj_id)
f.close()
