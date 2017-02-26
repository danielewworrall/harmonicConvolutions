'''Models'''

import os
import sys
import time

import numpy as np
import tensorflow as tf

import equivariant_loss as el

from spatial_transformer import transformer


def conv(x, shape, name='0', bias_init=0.01, return_params=False):
	"""Basic convolution"""
	He_initializer = tf.contrib.layers.variance_scaling_initializer()
	W = tf.get_variable('W'+name, shape=shape, initializer=He_initializer)
	z = tf.nn.conv2d(x, W, (1,1,1,1), 'SAME', name='conv'+str(name))
	return bias_add(z, shape[3], name=name)


def bias_add(x, nc, bias_init=0.01, name='0'):
	const_initializer = tf.constant_initializer(value=bias_init)
	b = tf.get_variable('b'+name, shape=nc, initializer=const_initializer)
	return tf.nn.bias_add(x, b)


def siamese_model(x, t_params, f_params, opt):
	"""Build siamese models for equivariance tests"""
	nc = opt['n_channels']
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('mouth') as scope:
		y1 = equivariant_mouth(x, nc, opt, name='_MC')
		# Transformer branch
		y_post = el.transform_features(y1, t_params, f_params)
		
		# Siamese loss
		x_pre = transformer(x, t_params, (xsh[1],xsh[2]))
		scope.reuse_variables()
		y2 = equivariant_mouth(x_pre, nc, opt, name='_MC')
		
		# Tail
	with tf.variable_scope('tail') as scope:
		y = tf.nn.max_pool(y1, (1,2,2,1), (1,2,2,1), padding='VALID')
		logits = invariant_tail(y, 2*nc, 3*nc, opt, name='tail')
	
	return logits, y_post, y2


def equivariant_mouth(x, nc, opt, name='_MC'):
	# L1
	l1 = build_mouth(x, 1, nc, opt, name='_M1'+name)
	l1 = tf.nn.max_pool(l1, (1,2,2,1), (1,2,2,1), padding='VALID')
	# L2
	return build_mouth(tf.nn.relu(l1), nc, 2*nc, opt, name='_M2'+name)


def build_mouth(x, ni, nc, opt, name='mouth'):
	"""Build the model we want"""
	l1 = conv(x, [3,3,ni,nc], name='1'+name )
	l1 = tf.nn.relu(l1)
	
	l2 = conv(l1, [3,3,nc,nc], name='2'+name)
	return l2


def invariant_tail(x, ni, nc, opt, name='tail'):
	"""Build the model we want"""
	l1 = conv(x, [3,3,ni,nc], name='1'+name )
	l1 = tf.nn.relu(l1)
	
	l2 = conv(l1, [3,3,nc,10], name='2'+name)
	l2 = tf.reduce_mean(l2, axis=[1,2])
	return l2