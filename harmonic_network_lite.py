"""
Harmonic Convolutions Lite

A simplified API for harmomin_network_ops
"""

import numpy as np
import tensorflow as tf

from harmonic_network_ops import *


def conv(x, out_shape, ksize, strides=(1,1,1,1), padding='VALID', phase=True,
			 max_order=1, stddev=0.4, name='lconv', device='/cpu:0'):
	"""Harmonic Convolution lite"""
	xsh = x.get_shape().as_list()
	shape = [ksize, ksize, xsh[5], out_shape]
	from harmonic_network_helpers import get_weights_dict, get_phase_dict
	Q = get_weights_dict(shape, max_order, std_mult=stddev, name='W'+name,
								device=device)
	P = None
	if phase == True:
		P = get_phase_dict(xsh[5], out_shape, max_order, name='P'+name,
								 device=device)
	W = get_filters(Q, filter_size=ksize, P=P)
	R = h_conv(x, W, strides=strides, padding=padding, max_order=max_order,
				  name=name)
	return R


def bn(x, train_phase, fnc=tf.nn.relu, decay=0.99, eps=1e-4, name='hbn',
		 device='/cpu:0'):
	"""Batch normalization for the magnitudes of X"""
	return h_batch_norm(x, fnc, train_phase, decay=decay, eps=eps, name=name,
							  device=device)


def nl(x, fnc=tf.nn.relu, eps=1e-4, name='nl', device='/cpu:0'):
	"""Alter nonlinearity for the complex domains"""
	return h_nonlin(x, fnc, eps=eps, name=name, device=device)


def mp(x, ksize=(1,1,1,1), strides=(1,1,1,1), name='mp'):
	"""Mean pooling"""
	with tf.name_scope(name) as scope:
		return mean_pooling(x, ksize=ksize, strides=strides)

def sum_mags(X, eps=1e-4, keep_dims=True):
	"""Sum the magnitudes of each of the complex feature maps in X.
	
	Output U = sum_i |X_i|
	
	X: dict of channels {rotation order: (real, imaginary)}
	eps: regularization since grad |Z| is infinite at zero (default 1e-4)
	"""
	return sum_magnitudes(X, eps, keep_dims=True)

def res(x, out_shape, ksize, depth, train_phase, fnc=tf.nn.relu, max_order=1,
		  phase=True, name='res', device='/cpu:0'):
	"""Residual block"""
	with tf.name_scope(name) as scope:
		y = x
		for i in xrange(depth):
			y = conv(y, out_shape, ksize, padding='SAME', phase=phase,
				  max_order=max_order, name=name+'_c'+str(i), device=device)
			if i == (depth-1):
				fnc = (lambda x: x)
			y = bn(y, train_phase, fnc=fnc, name=name+'_nl'+str(i), device=device)
		xsh = x.get_shape().as_list()
		ysh = y.get_shape().as_list()
		x = tf.pad(x, [[0,0],[0,0],[0,0],[0,0],[0,0],[0,ysh[5]-xsh[5]]])
		return y + x




































