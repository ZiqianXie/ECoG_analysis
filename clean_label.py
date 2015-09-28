import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.cluster import KMeans

def nonzero_seg(x):
	'''find start and end indices of nonzero segments of a 1D array'''
	nzid = x.nonzero()[0]
	tmp = np.where(np.diff(nzid) != 1)[0]
	return np.vstack((np.hstack((nzid[0],nzid[tmp +1])),
	np.hstack((nzid[tmp],nzid[-1])))).T

def peak(x, idx):
	'''test if x[idx] is a peak'''
	if idx==0 or idx==len(x):
		return False
	if (x[idx]>=x[idx-1] and x[idx]>=x[idx+1])^\
		(x[idx]==x[idx-1] and x[idx]==x[idx+1]):
		return True
	else:
		return False

	

def flat(x, idx, valleyidx = None):
	'''
	Flatten a 1D signal peak, find two valley points
	surrounding the peak and make the segment between two
	valley points zero
	'''
	if not peak(x, idx):
		return valleyidx
	if valleyidx is None:
		l1 = np.logical_and(x[1:-1]<=x[0:-2],x[1:-1]<=x[2:])
		l2 = np.logical_and(x[1:-1]==x[0:-2],x[1:-1]==x[2:])
		valleyidx = np.where(np.logical_xor(l1,l2))[0]+1
	v1 = valleyidx[valleyidx<idx]
	v2 = valleyidx[valleyidx>idx]
	if v1.size > 0:
		p0 = v1[-1]
	else:
		p0 = idx - 1
	if v2.size >0:
		p1 = v2[0]
	else:
		p1 = idx + 1
	x[p0:p1+1] = 0
	return valleyidx

def clean_label1D(x):
	'''
	Find all peaks of a 1D signal,
	iteratively run two cluster KMean on all peaks, 
	flatten small ones.
	'''
        x -= x.min()
	x_old = np.zeros_like(x)
	x_new = x
	while not np.allclose(x_new, x_old):
		peakidx = np.where(np.logical_and(x[1:-1]>=x[0:-2],
			x[1:-1]>=x[2:]))[0] + 1
		km = KMeans(n_clusters = 2)
		km.fit(x[peakidx].reshape(-1,1))
		min_cluster = np.argmin(km.cluster_centers_)
		minidx = np.where(km.predict(x[peakidx].reshape(-1,1))==min_cluster)[0]
		x_old = x_new.copy()
		valleyidx = None
		for idx in minidx:
			valleyidx = flat(x_new, peakidx[idx], valleyidx)
	x_new[[0,-1]] = 0
        nzseg = nonzero_seg(x_new)
        for seg in nzseg:
            x_new[seg[0]:seg[1]+1] -= x[seg[0]-1]
            x_new[x_new<0] = 0
	return x_new

def clean_label(x):
	'''
	Clean x by column, max out and clean again.
	'''
	x = x.copy()
	np.apply_along_axis(clean_label1D, 0, x)
	vmax = x.max(1)
	for column in x.T:
		column[column!=vmax] = 0
	np.apply_along_axis(clean_label1D, 0, x)
	return x

if __name__ == '__main__':
    import h5py
    f = h5py.File('ECoG_data.h5', 'r+')
    for subject in f:
        for d in ['train', 'test']:
            tmp = f[subject][d+'_dg'][...]
            new = 'cleaned_'+d+'_dg'
            if new not in f[subject]:
                f[subject].create_dataset(new, data
					=clean_label(tmp))
    f.close()
