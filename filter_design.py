# -*- coding: utf-8 -*-
import glob
import h5py


def filter_design():
    if not glob.glob('ECoG_filter.h5'):
        from scipy.signal import remez
        f1 = remez(121, [0, 8, 12, 20, 24, 500], [0, 1, 0], Hz=1000)
        f2 = remez(121, [0, 50, 60, 100, 110, 500], [0, 1, 0], Hz=1000)
        f3 = remez(121, [0, 90, 100, 200, 210, 500], [0, 1, 0], Hz=1000)
        filters = [f1, f2, f3]
        with h5py.File('ECoG_filter.h5', 'w') as f:
            f.create_dataset('filters', data=filters)
    else:
        with h5py.File('ECoG_filter.h5', 'r') as f:
            filters = f['filters'][:]
    return filters
