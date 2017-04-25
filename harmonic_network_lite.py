"""
Harmonic Convolutions Lite

A simplified API for harmomin_network_ops
"""

import numpy as np
import tensorflow as tf

from harmonic_network_ops import *


def conv2d(x, n_channels, ksize, strides=(1,1,1,1), padding='VALID', phase=True,
             max_order=1, stddev=0.4, n_rings=None, name='conv2d'):
    """Harmonic Convolution lite

    x: input tf tensor, shape [batchsize,height,width,order,complex,channels],
    e.g. a real input tensor of rotation order 0 could have shape
    [16,32,32,3,1,9], or a complex input tensor of rotation orders 0,1,2, could
    have shape [32,121,121,3,2,10]
    n_channels: number of output channels (int)
    ksize: size of square filter (int)
    strides: stride size (4-tuple: default (1,1,1,1))
    padding: SAME or VALID (defult VALID)
    phase: use a per-channel phase offset (default True)
    max_order: maximum rotation order e.g. max_order=2 uses 0,1,2 (default 1)
    stddev: scale of filter initialization wrt He initialization
    name: (default 'lconv')
    device: (default '/cpu:0')
    """
    xsh = x.get_shape().as_list()
    shape = [ksize, ksize, xsh[5], n_channels]
    Q = get_weights_dict(shape, max_order, std_mult=stddev, n_rings=n_rings, name='W'+name)
    if phase == True:
        P = get_phase_dict(xsh[5], n_channels, max_order, name='phase'+name)
    else:
        P = None
    W = get_filters(Q, filter_size=ksize, P=P, n_rings=n_rings)
    R = h_conv(x, W, strides=strides, padding=padding, max_order=max_order,
               name=name)
    return R


def batch_norm(x, train_phase, fnc=tf.nn.relu, decay=0.99, eps=1e-4, name='hbn'):
    """Batch normalization for the magnitudes of X"""
    return h_batch_norm(x, fnc, train_phase, decay=decay, eps=eps, name=name)


def non_linearity(x, fnc=tf.nn.relu, eps=1e-4, name='nl'):
    """Alter nonlinearity for the complex domains"""
    return h_nonlin(x, fnc, eps=eps, name=name)


def mean_pool(x, ksize=(1,1,1,1), strides=(1,1,1,1), name='mp'):
    """Mean pooling"""
    with tf.name_scope(name) as scope:
        return mean_pooling(x, ksize=ksize, strides=strides)


def sum_magnitudes(x, eps=1e-12, keep_dims=True):
    """Sum the magnitudes of each of the complex feature maps in X.

    Output U = sum_i |x_i|

    x: input tf tensor, shape [batchsize,height,width,channels,complex,order],
    e.g. a real input tensor of rotation order 0 could have shape
    [16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
    have shape [32,121,121,32,2,3]
    eps: regularization since grad |x| is infinite at zero (default 1e-4)
    keep_dims: whether to collapse summed dimensions (default True)
    """
    R = tf.reduce_sum(tf.square(x), axis=[4], keep_dims=keep_dims)
    return tf.reduce_sum(tf.sqrt(tf.maximum(R,eps)), axis=[3], keep_dims=keep_dims)


def stack_magnitudes(X, eps=1e-12, keep_dims=True):
    """Stack the magnitudes of each of the complex feature maps in X.

    Output U = concat(|X_i|)

    X: dict of channels {rotation order: (real, imaginary)}
    eps: regularization since grad |Z| is infinite at zero (default 1e-12)
    """
    R = tf.reduce_sum(tf.square(X), axis=[4], keep_dims=keep_dims)
    return tf.sqrt(tf.maximum(R,eps))
