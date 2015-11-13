# -*- coding: utf-8 -*-
import theano.tensor as T
import numpy as np
import lasagne.layers as layers
from lasagne.layers import get_output
from lasagne import regularization
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne.updates import adam
from custom_layers import CustomPoolLayer
from lasagne.objectives import squared_error, aggregate
from ECoG_model import striding
import h5py
import cPickle as pickle


def objective(layers,
              loss_function,
              target,
              aggregate=aggregate,
              deterministic=False,
              l1=0,
              l2=0,
              tv=0,
              get_output_kw=None):
    if get_output_kw is None:
        get_output_kw = {}
    output_layer = layers[-1]
    network_output = get_output(
        output_layer, deterministic=deterministic, **get_output_kw)
    loss = aggregate(loss_function(network_output, target))

    if l1:
        loss += regularization.regularize_layer_params(
            layers[-2], regularization.l1) * l1
    if l2:
        loss += regularization.regularize_layer_params(
            layers[-2], regularization.l2) * l2
    if tv:
        loss += T.mean(T.abs_(network_output[:, 1:] -
                              network_output[:, :-1]))*tv
    return loss


if __name__ == "__main__":
    pooling_size = 40
    t1 = 7
    t2 = 5
    subj = 'sub1'
    finger = 1
    with h5py.File('ECoG_data.h5', 'r') as f:
        u = f['sub1/unmixing_matrix'][:]
    with h5py.File('selected_channel.h5', 'r') as f:
        c = f['selected_channel'][:]
    u = u[:, c].astype('f')
    s_num = u.shape[1]
    with h5py.File('ECoG_filter.h5', 'r') as f:
        filters = f['filters'][:].astype('f')
    filters = filters[:, np.newaxis, np.newaxis, :]
    filter_len = filters.shape[-1]
    input_size = pooling_size * (t2 + t1 - 1) + filter_len - 1
    linear_model = pickle.load(open('linear_model_sub1_1', 'rb'))
    W0 = linear_model.coef_.reshape(62, 3, 7)[c, :, :].astype('f')
    W0 = W0.transpose(1, 0, 2)
    W0 = W0[np.newaxis, :, ::-1, ::-1]
    b0 = np.array([linear_model.intercept_], dtype='f')
    model1 = [(layers.InputLayer, {'shape': (None, 62, 1, input_size)}),
              (layers.NINLayer, {'num_units': s_num, 'W': u, 'b': None,
                                 'nonlinearity': None}),
              (layers.DimshuffleLayer, {'pattern': (0, 2, 1, 3)}),
              (layers.Conv2DLayer, {'num_filters': 3,
                                    'filter_size': (1, filter_len),
                                    'W': filters, 'b': None,
                                    'nonlinearity': None}),
              (CustomPoolLayer, {'pool_size': pooling_size}),
              (layers.Conv2DLayer, {'num_filters': 1,
                                    'filter_size': (s_num, t1),
                                    'W': W0,
                                    'b': b0,
                                    }),
              (layers.ReshapeLayer, {'shape':  ([0], t2)})]
    net = NeuralNet(layers=model1,
                    max_epochs=100,
                    update=adam,
                    update_learning_rate=1e-9,
                    objective=objective,
                    objective_loss_function=squared_error,
                    objective_tv=0,
                    objective_l1=0.1,
                    objective_l2=0,
                    batch_iterator_train=BatchIterator(batch_size=128),
                    batch_iterator_test=BatchIterator(batch_size=128),
                    verbose=1,
                    regression=True,
                    )

    with h5py.File('ECoG_data.h5', 'r') as f:
        X = f[subj]['train_data'][:]
        mean = X.mean(0)
        X = (X-mean)
        y = f[subj]['cleaned_train_dg'][:, finger]
        Xt = f[subj]['test_data'][:]
        tmean = Xt.mean(0)
        Xt = (Xt-tmean)
        yt = f[subj]['cleaned_test_dg'][:, finger]
    X = striding(X, input_size, pooling_size)[:, :, np.newaxis, :].astype('f')
    y = striding(y[:, np.newaxis], t2+t1-1, 1)[filter_len/pooling_size:,
                                               0, t1-1:].astype('f')
    Xt = striding(Xt,
                  input_size, pooling_size)[:, :, np.newaxis, :].astype('f')
    yt = striding(yt[:, np.newaxis], t2+t1-1, 1)[filter_len/pooling_size:,
                                                 0, t1-1:].astype('f')
    net.fit(X, y)
