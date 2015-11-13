import numpy as np
import theano.tensor as T
import lasagne.layers as layers
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne.objectives import squared_error
from lasagne.updates import adam
from lasagne.nonlinearities import sigmoid
from custom_layers import CustomPoolLayer
from ECoG_model import striding
import h5py
import cPickle as pickle
from model1 import objective


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
    logistic_model = pickle.load(open('logistic_model_sub1_1', 'rb'))
    W1 = logistic_model.coef_.reshape(s_num, 3, 7).astype('f')
    W1 = W1.transpose(1, 0, 2)
    W1 = W1[np.newaxis, :, ::-1, ::-1]
    b1 = logistic_model.intercept_.astype('f')
    model2 = [(layers.InputLayer, {'shape': (None, 62, 1, input_size)}),
              (layers.NINLayer, {'num_units': s_num, 'W': u, 'b': None,
                                 'nonlinearity': None}),
              (layers.DimshuffleLayer, {'pattern': (0, 2, 1, 3)}),
              (layers.Conv2DLayer, {'num_filters': 3,
                                    'filter_size': (1, filter_len),
                                    'W': filters, 'b': None,
                                    'nonlinearity': None}),
              (CustomPoolLayer, {'pool_size': pooling_size,
                                 'name': 'custompool4'}),
              (layers.Conv2DLayer, {'num_filters': 1,
                                    'filter_size': (s_num, t1),
                                    'W': W0,
                                    'b': b0,
                                    'name': 'conv2d5'
                                    }),
              (layers.Conv2DLayer, {'num_filters': 1,
                                    'filter_size': (s_num, t1),
                                    'W': W1,
                                    'b': b1,
                                    'nonlinearity': sigmoid,
                                    'name': 'conv2d6',
                                    'incoming': 'custompool4'
                                    }),
              (layers.ElemwiseMergeLayer, {'incomings': ['conv2d5', 'conv2d6'],
                                           'merge_function': T.mul}),
              (layers.ReshapeLayer, {'shape':  ([0], t2)})]
    net = NeuralNet(layers=model2,
                    max_epochs=100,
                    update=adam,
                    update_learning_rate=1e-9,
                    objective=objective,
                    objective_loss_function=squared_error,
                    objective_tv=1,
                    objective_l1=0.1,
                    objective_l2=0.1,
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
