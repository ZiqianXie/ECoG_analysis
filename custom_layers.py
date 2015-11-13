# -*- coding: utf-8 -*-
import lasagne.layers as layers
import theano.tensor as T


class CustomPoolLayer(layers.Layer):
    def __init__(self, incoming, pool_size, axis=3, **kwargs):
        super(CustomPoolLayer, self).__init__(incoming, **kwargs)
        self.pool_size = pool_size
        self.axis = axis
        self.pool_function = lambda x, axis: T.log(T.sum(x**2, axis))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # make a mutable copy
        if output_shape[self.axis] is None:
            return output_shape
        output_shape[self.axis] = input_shape[self.axis] // self.pool_size
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_shape = tuple(input.shape)
        num_feature_maps = input_shape[self.axis]
        num_feature_maps_out = num_feature_maps // self.pool_size

        pool_shape = (input_shape[:self.axis] +
                      (num_feature_maps_out, self.pool_size) +
                      input_shape[self.axis+1:])

        input_reshaped = input.reshape(pool_shape)
        return self.pool_function(input_reshaped, axis=self.axis + 1)
