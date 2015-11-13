# -*- coding: utf-8 -*-
# from lasagne.objectives import squared_error
# from lasagne.updates import sgd
import lasagne
import lasagne.layers as layers
import theano.tensor as T
from custom_layers import CustomPoolLayer
from theano import function
import numpy as np
import cPickle as pickle
import h5py


with h5py.File('ECoG_data.h5', 'r') as f:
    u = f['sub1/unmixing_matrix'][:]
with h5py.File('selected_channel.h5', 'r') as f:
    c = f['selected_channel'][:]
u = u[:, c].astype('f')
s_num = u.shape[1]
with h5py.File('ECoG_filter.h5', 'r') as f:
    filters = f['filters'][:].astype('f')
filters = filters[:, np.newaxis, np.newaxis, :]
linear_model = pickle.load(open('linear_model_sub1_1', 'rb'))
W0 = linear_model.coef_.reshape(62, 3, 7)[c, :, :].astype('f')
W0 = W0.transpose(1, 0, 2)
W0 = W0[np.newaxis, :, ::-1, ::-1]
b0 = np.array([linear_model.intercept_], dtype='f')
logistic_model = pickle.load(open('logistic_model_sub1_1', 'rb'))
W1 = logistic_model.coef_.reshape(s_num, 3, 7).astype('f')
W1 = W1.transpose(1, 0, 2)
W1 = W1[np.newaxis, :, ::-1, ::-1]
b1 = logistic_model.intercept_.astype('f')
filter_len = 121
pooling_size = 40
t1 = 7
t2 = 5

input_size = pooling_size * (t2 + t1 - 1) + filter_len - 1
x = T.tensor4()
t = T.matrix()
l_in = layers.InputLayer((None, 62, 1, input_size), input_var=x)
l_sf = layers.NINLayer(l_in, num_units=s_num,
                       W=u,
                       b=None, nonlinearity=None)
l_ds = layers.DimshuffleLayer(l_sf, (0, 2, 1, 3))
l_conv1 = layers.Conv2DLayer(l_ds, num_filters=3, filter_size=(1, filter_len),
                             b=None,
                             W=filters,
                             nonlinearity=None)
l_pool = CustomPoolLayer(l_conv1, pooling_size)
l_conv2 = layers.Conv2DLayer(l_pool, num_filters=1, filter_size=(s_num, t1),
                             W=W0,
                             b=b0, nonlinearity=None)
l_conv3 = layers.Conv2DLayer(l_pool, num_filters=1, filter_size=(s_num, t1),
                             W=W1,
                             b=b1, nonlinearity=lasagne.nonlinearities.sigmoid)
l_merge = layers.ElemwiseMergeLayer([l_conv2, l_conv3], T.mul)
l_rs = layers.ReshapeLayer(l_merge, ([0], -1))
y0 = layers.get_output(l_rs)
fun = function([x], y0)
'''
loss = squared_error(y, t) + 1e-2*T.sum(T.abs_(y[:, 1:] - y[:, :-1]))
loss = loss.mean()
p = layers.get_all_param_values(l_rs, trainable=True)
updates = sgd(loss, layers.get_all_params(l_rs, trainable=True),
              learning_rate=0.001)
f = function([x, t], T.grad(loss, layers.get_all_params(l_rs, trainable=True)))
a = np.random.randn(10, 62, 1, 5480).astype('f')
b = np.random.randn(10, 128).astype('f')
r = f(a, b)
'''
