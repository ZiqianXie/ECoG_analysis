# -*- coding: utf-8 -*-
import lasagne.layers as layers
import theano.tensor as T
from theano import function

x = T.tensor4()
l_in = layers.InputLayer((1, 2, 3, 4), input_var=x)
l_conv = layers.Conv2DLayer(l_in, num_filters=2, filter_size=(3, 4), b=None,
                            nonlinearity=None)
y = layers.get_output(l_conv)
fun = function([x], y)
g = function([], l_conv.get_params())
