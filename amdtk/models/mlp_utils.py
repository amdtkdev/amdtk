
"""
Utilities for MLP object.

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import abc
import logging
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool


# Create the module's logger.
logger = logging.getLogger(__name__)


def _linear(x):
    """Linear activation. Do nothing on the input."""
    return x


# Possible activation for the hidden units.
ACTIVATIONS = {
    'softmax': T.nnet.softmax,
    'sigmoid': T.nnet.sigmoid,
    'tanh': T.tanh,
    'relu': T.nnet.relu,
    'linear': _linear
}


class MLPError(Exception):
    """Base class for exceptions in this module."""
    pass


class UnkownActivationError(MLPError):
    """Raised when the given activation is not known."""

    def __init__(self, activation):
        self.activation = str(activation)

    def __str__(self):
        return '"' + self.activation + '" is not one of the pre-defined " \
               "activations: "' + '", "'.join(ACTIVATIONS.keys()) + '"'


def _init_weights_matrix(dim_in, dim_out, activation, borrow=True):
    val = np.sqrt(6. / (dim_in + dim_out))
    if activation == 'sigmoid':
        retval = 4 * np.random.uniform(low=-val, high=val,
                                       size=(dim_in, dim_out))
    elif activation == 'tanh':
        retval = np.random.uniform(low=-val, high=val,
                                   size=(dim_in, dim_out))
    elif (activation == 'relu' or activation == 'linear' or
         activation == 'softmax'):
        retval = np.random.normal(0., 0.01, size=(dim_in, dim_out))
    else:
        raise UnkownActivationError(activation)

    return theano.shared(np.asarray(retval, dtype=theano.config.floatX),
                         borrow=borrow)


def init_residual_weights_matrix(dim_in, dim_out, borrow=True):
    """Partial isometry initialization."""
    if dim_out == dim_in:
        weights = np.identity(dim_in)
    else:
        d = max(dim_in, dim_out)
        weights = np.linalg.qr(np.random.randn(d,d))[0][:dim_in,:dim_out]
    return theano.shared(np.asarray(weights, dtype=theano.config.floatX),
                         borrow=borrow)


def _init_bias(dim, borrow=True):
    return theano.shared(np.zeros(dim, dtype=theano.config.floatX) + .01,
                                  borrow=borrow)


class LogisticRegressionLayer(object):

    def __init__(self, inputs, dim_in, dim_out, activation):
        self.inputs = inputs.flatten(2)
        self.dim_in = dim_in
        self.dim_out = dim_out
        weights = _init_weights_matrix(dim_in, dim_out, activation)
        bias = _init_bias(dim_out)
        self.outputs = ACTIVATIONS[activation](T.dot(
            self.inputs, weights) + bias)
        self.params = [weights, bias]

class StdLayer(object):

    def __init__(self, inputs, dim_in, dim_out, activation):
        if inputs is None:
            self.inputs = T.matrix(dtype=theano.config.floatX)
        else:
            self.inputs = inputs.flatten(2)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        weights = _init_weights_matrix(dim_in, dim_out, activation)
        bias = _init_bias(dim_out)
        self.outputs = ACTIVATIONS[activation](
            T.dot(self.inputs, weights) + bias)
        self.params = [weights, bias]


class GaussianLayer(object):

    def __init__(self, inputs, dim_in, dim_out, activation):
        if inputs is None:
            self.inputs = T.matrix(dtype=theano.config.floatX)
        else:
            self.inputs = inputs.flatten(2)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        shared_layer = StdLayer(inputs, dim_in, 2 * dim_out, activation)

        self.mean, raw_logvar = \
            theano.tensor.split(shared_layer.outputs, [dim_out, dim_out], 2,
                                axis=-1)
        #self.var = T.log(1 + T.exp(raw_logvar))
        self.var = T.exp(raw_logvar)

        self.params = shared_layer.params
        self.outputs = self.mean


class ConvLayer(object):

    def __init__(self, inputs, fmap_dim_in, fmap_dim_out, f_height, f_width,
                activation):
        if inputs is None:
            self.inputs = T.tensor4(dtype=theano.config.floatX)
        else:
            self.inputs = inputs.flatten(4)
        self.fmap_dim_in = fmap_dim_in
        self.fmap_dim_out = fmap_dim_out
        self.f_height = f_height
        self.f_width = f_width
        self.activation = activation

        w_bound = np.sqrt(fmap_dim_in * f_height * f_width)
        weights = theano.shared(
            np.asarray(
                np.random.uniform(
                    low=-1.0 / w_bound,
                    high=1.0 / w_bound,
                    size=(fmap_dim_out, fmap_dim_in, f_height, f_width)
                ),
                dtype=self.inputs.dtype
            )
        )

        bias = theano.shared(
            np.zeros(
                (fmap_dim_out,),
                dtype=self.inputs.dtype
            )
        )

        out = T.nnet.conv2d(self.inputs, weights, border_mode='half')
        out += bias.dimshuffle('x', 0, 'x', 'x')
        self.outputs = ACTIVATIONS[activation](out)

        self.params = [weights, bias]


class PoolLayer(object):

    def __init__(self, inputs, maxpool_height, maxpool_width):
        if inputs is None:
            self.inputs = T.tensor4(dtype=theano.config.floatX)
        else:
            self.inputs = inputs.flatten(4)
        self.maxpool_height = maxpool_height
        self.maxpool_width = maxpool_width
        self.outputs = pool.pool_2d(
            self.inputs,
            (maxpool_height, maxpool_width),
            ignore_border=True
        )

        # Pooling layer has no learnable parameters.
        self.params = []


# Possible layer types.
LAYER_TYPES = {
   'standard': StdLayer,
   'convolution': ConvLayer,
   'pool': PoolLayer,
   'gaussian': GaussianLayer
}


class NeuralNetwork(object):

    def __init__(self, structure, inputs=None):
        # Build the neural network.
        self.layers = []
        self.params = []
        current_inputs = inputs
        for layer_struct in structure:
            layer_type = layer_struct[0]
            layer_params = layer_struct[1:]

            logger.debug('create nnet layer type={layer_type} '
                'params={params}'.format(layer_type=layer_type,
                                         params=layer_params))

            self.layers.append(
                LAYER_TYPES[layer_type](
                    current_inputs,
                    *layer_params
                )
            )
            self.params += self.layers[-1].params
            current_inputs = self.layers[-1].outputs

        self.inputs = self.layers[0].inputs
        self.outputs = self.layers[-1].outputs


class GaussianNeuralNetwork(NeuralNetwork):

    def __init__(self, structure, inputs=None, n_samples=10):
        NeuralNetwork.__init__(self, structure, inputs)
        self.mean = self.layers[-1].outputs
        self.var = self.layers[-1].var
        self.n_samples = n_samples

        # Noise variable for the reparameterization trick.
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(123)
        else:
            srng = T.shared_randomstreams.RandomStreams()
            self.eps = srng.normal((n_samples, self.mean.shape[0],
                                    self.mean.shape[1]))

        # Latent variable.
        self.sample = self.mean + T.sqrt(self.var) * self.eps

        self.sample = T.reshape(self.sample,
            (n_samples * self.mean.shape[0], -1))

        # Build the functions.
        self.forward = theano.function(
            inputs=[self.inputs],
            outputs=[self.mean, self.var]
        )

