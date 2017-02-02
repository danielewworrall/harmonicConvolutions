"""
Harmonic Convolutions Lite

A simplified API for harmomin_network_ops
"""

import numpy as np
import tensorflow as tf

from harmonic_network_ops import *

def conv2d(x, n_channels, ksize, strides=(1,1,1,1), padding='VALID', phase=True,
			 max_order=1, stddev=0.4, n_rings=None, name='lconv', device='/cpu:0'):
	"""Harmonic Convolution lite
	
	x: input tf tensor, shape [batchsize,height,width,order,complex,channels],
	e.g. a real input tensor of rotation order 0 could have shape
	[16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
	have shape [32,121,121,32,2,3]
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
	Q = get_weights_dict(shape, max_order, std_mult=stddev, n_rings=n_rings,
								name='W'+name, device=device)
	if phase == True:
		P = get_phase_dict(xsh[5], n_channels, max_order, name='P'+name,
								 device=device)
	W = get_filters(Q, filter_size=ksize, P=P, n_rings=n_rings)
	R = h_conv(x, W, strides=strides, padding=padding, max_order=max_order,
				  name=name)
	return R


def batch_norm(x, is_training, fnc=tf.nn.relu, decay=0.99, eps=1e-12, name='hbn',
		 device='/cpu:0'):
	"""Batch normalization for the magnitudes of X
	
	x: input tf tensor, shape [batchsize,height,width,order,complex,channels],
	e.g. a real input tensor of rotation order 0 could have shape
	[16,32,32,1,1,3], or a complex input tensor of rotation orders 0,1,2, could
	have shape [32,121,121,3,2,32]
	is_training: tf bool indicating training status
	fnc: nonlinearity applied to magnitudes (default tf.nn.relu)
	decay: exponential decay rate of statistics trackers
	eps: regularization for estimation of magnitudes
	name: (default 'hbn')
	device: (default '/cpu:0')
	"""
	return h_batch_norm(x, fnc, is_training, decay=decay, eps=eps, name=name,
							  device=device)


def log_batch_norm(x, is_training, fnc=tf.nn.relu, decay=0.99, eps=1e-12, name='hbn',
		 device='/cpu:0'):
	"""Batch normalization on the logarithm of the activations"""
	y = h_batch_norm(tf.log(x), fnc, is_training, decay=decay, eps=eps,
						  name=name, device=device)
	return tf.exp(y)


def nonlinearity(x, fnc=tf.nn.relu, eps=1e-12, name='nl', device='/cpu:0'):
	"""Alter nonlinearity for the complex domain
	
	x: input tf tensor, shape [batchsize,height,width,channels,complex,order],
	e.g. a real input tensor of rotation order 0 could have shape
	[16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
	have shape [32,121,121,32,2,3]
	fnc: nonlinearity applied to magnitudes (default tf.nn.relu)
	eps: regularization since grad |x| is infinite at zero
	name: (default 'nl')
	device: (default '/cpu:0')
	"""
	return h_nonlin(x, fnc, eps=eps, name=name, device=device)


def mean_pool(x, ksize=(1,1,1,1), strides=(1,1,1,1), name='mp'):
	"""Mean pooling
	
	x: input tf tensor, shape [batchsize,height,width,channels,complex,order],
	e.g. a real input tensor of rotation order 0 could have shape
	[16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
	have shape [32,121,121,32,2,3]
	ksize: size of square filter (int)
	strides: stride size (4-tuple: default (1,1,1,1))
	name: (default 'mp')
	"""
	with tf.name_scope(name) as scope:
		return mean_pooling(x, ksize=ksize, strides=strides)
	

def max_pool(x, ksize=(1,1,1,1), strides=(1,1,1,1), name='mxp'):
	"""Mean-max pooling
	
	x: input tf tensor, shape [batchsize,height,width,channels,complex,order],
	e.g. a real input tensor of rotation order 0 could have shape
	[16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
	have shape [32,121,121,32,2,3]
	ksize: size of square filter (int)
	strides: stride size (4-tuple: default (1,1,1,1))
	name: (default 'mp')
	"""
	with tf.name_scope(name) as scope:
		return max_pooling(x, ksize=ksize, strides=strides)
	

def mean_max_pool(x, ksize=(1,1,1,1), strides=(1,1,1,1), name='mxp'):
	"""Mean-max pooling
	
	x: input tf tensor, shape [batchsize,height,width,channels,complex,order],
	e.g. a real input tensor of rotation order 0 could have shape
	[16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
	have shape [32,121,121,32,2,3]
	ksize: size of square filter (int)
	strides: stride size (4-tuple: default (1,1,1,1))
	name: (default 'mp')
	"""
	with tf.name_scope(name) as scope:
		return mean_max_pooling(x, ksize=ksize, strides=strides)


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
	R = tf.reduce_sum(tf.square(x), reduction_indices=[4], keep_dims=keep_dims)
	return tf.sqrt(R + eps)


def residual_block(x, n_channels, ksize, depth, is_training, fnc=tf.nn.relu,
						 stddev=0.4, max_order=1, phase=True, name='res',
						 device='/cpu:0'):
	"""Harmonic version of a residual block
	
	x: input tf tensor, shape [batchsize,height,width,channels,complex,order],
	e.g. a real input tensor of rotation order 0 could have shape
	[16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
	have shape [32,121,121,32,2,3]
	n_channels: number of output channels (int)
	ksize: size of square filter (int)
	depth: number of convolutions per block
	is_training: tf bool indicating training status
	fnc: nonlinearity applied to magnitudes (default tf.nn.relu)
	max_order: maximum rotation order e.g. max_order=2 uses 0,1,2 (default 1)
	phase: use a per-channel phase offset (default True)
	name: (default 'res')
	device: (default '/cpu:0')
	"""
	with tf.name_scope(name) as scope:
		y = x
		for i in xrange(depth):
			y = conv2d(y, n_channels, ksize, padding='SAME', phase=phase,
				  max_order=max_order, name=name+'_c'+str(i), device=device,
				  stddev=stddev)
			if i == (depth-1):
				fnc = (lambda x: x)
			y = batch_norm(y, is_training, fnc=fnc, name=name+'_nl'+str(i),
								device=device)
		xsh = x.get_shape().as_list()
		ysh = y.get_shape().as_list()
		x = tf.pad(x, [[0,0],[0,0],[0,0],[0,0],[0,0],[0,ysh[5]-xsh[5]]])
		return y + x









