# -*- coding: utf-8 -*-

import lasagne
import lasagne.layers as layers
from lasagne.objectives import squared_error, aggregate
from lasagne import regularization
import theano.tensor as T
from theano import function


def objective(output_layer,
              regularize_layers,
              target,
              loss_function=squared_error,
              aggregate=aggregate,
              deterministic=False,
              l1=0,
              l2=0,
              tv=0):
    network_output = layers.get_output(
        output_layer, deterministic=deterministic)
    loss = aggregate(loss_function(network_output, target))
    for layer in regularize_layers:
        if l1:
            loss +=\
                regularization.regularize_layer_params(layer,
                                                       regularization.l1) * l1
        if l2:
            loss +=\
                regularization.regularize_layer_params(layer,
                                                       regularization.l2) * l2
    if tv:
        loss += T.mean(T.abs_(network_output[:, 1:] -
                              network_output[:, :-1]))*tv
    return loss


filter_len = 121
pooling_size = 40
t1 = 7
t2 = 5
s_num = 19
input_size = pooling_size * (t2 + t1 - 1) + filter_len - 1
x = T.tensor4()
l_in = layers.InputLayer((None, 62, 1, input_size), input_var=x)
l_sf = layers.NINLayer(l_in, num_units=s_num,
                       b=None, nonlinearity=None)
l_ds = layers.DimshuffleLayer(l_sf, (0, 2, 1, 3))
l_conv1 = layers.Conv2DLayer(l_ds, num_filters=3, filter_size=(1, filter_len),
                             b=None,
                             nonlinearity=None)
l_pool = layers.FeaturePoolLayer(l_conv1, 40, 3,
                                 lambda x, axis: T.log(T.sum(x**2, axis)))
l_conv2 = layers.Conv2DLayer(l_pool, num_filters=1, filter_size=(s_num, t1),
                             nonlinearity=None)
l_conv3 = layers.Conv2DLayer(l_pool, num_filters=1, filter_size=(s_num, t1),
                             nonlinearity=lasagne.nonlinearities.sigmoid)
l_merge = layers.ElemwiseMergeLayer([l_conv2, l_conv3], T.mul)
l_rs = layers.ReshapeLayer(l_merge, ([0], -1))
regularize_layers = [l_conv2, l_conv3]
t = T.matrix()
params = layers.get_all_params(l_rs, trainable=True)
loss = objective(l_rs, regularize_layers, t)
updates = lasagne.updates.sgd(loss, params, learning_rate=1e-4)
train_fn = function([x, t], loss, updates=updates)
# # # # # # # # #
# t = T.tensor4()
# params = layers.get_all_params(l_merge, trainable=True)
# loss = objective(l_merge, regularize_layers, t)
# updates = lasagne.updates.sgd(loss, params, learning_rate=1e-4)
# train_fn = function([x, t], loss, updates=updates)
