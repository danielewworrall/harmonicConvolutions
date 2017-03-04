'''Models'''

import os
import sys
import time

import numpy as np
import tensorflow as tf

import equivariant_loss as el

from spatial_transformer import transformer


def conv(x, shape, opt, name='conv', bias_init=0.01):
	"""Basic convolution"""
	name = 'conv_{:s}'.format(name)
	He_initializer = tf.contrib.layers.variance_scaling_initializer()
	l2_regularizer = tf.contrib.layers.l2_regularizer(opt['weight_decay'])
	W = tf.get_variable(name+'_W', shape=shape, initializer=He_initializer,
							  regularizer=l2_regularizer)
	z = tf.nn.conv2d(x, W, (1,1,1,1), 'SAME', name='conv'+str(name))
	return bias_add(z, shape[3], name=name+'_b')


def bias_add(x, nc, bias_init=0.01, name='b'):
	const_initializer = tf.constant_initializer(value=bias_init)
	b = tf.get_variable(name, shape=nc, initializer=const_initializer)
	return tf.nn.bias_add(x, b)


def siamese_model(x, is_training, t_params, f_params, opt):
	"""Build siamese models for equivariance tests"""
	nc = opt['n_channels']
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('siamese') as scope:
		y = VGG(x, is_training, opt, branch=1)
		# Transformer branch
		y1 = []
		for features in y[:-1]:
			y1.append(el.transform_features(features, t_params, f_params))
		# Siamese loss
		x_pre = transformer(x, t_params, (xsh[1],xsh[2]))
		scope.reuse_variables()
		y2 = VGG(x_pre, is_training, opt, branch=2)
		
		# Tail
	with tf.variable_scope('tail') as scope:
		gap = tf.reduce_mean(y[-1], axis=(1,2), keep_dims=True)
		logits = conv(gap, [1,1,8*nc,opt['n_labels']], opt, name='VGGout')
		logits = tf.squeeze(logits, squeeze_dims=(1,2))
	
	return logits, y1, y2[:-1]


def single_model(x, is_training, opt):
	"""Build siamese models for equivariance tests"""
	nc = opt['n_channels']
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('siamese') as scope:
		y = VGG(x, is_training, opt)
				
		# Tail
	with tf.variable_scope('tail') as scope:
		gap = tf.reduce_mean(y[-1], axis=(1,2), keep_dims=True)
		logits = conv(tf.nn.relu(gap), [1,1,8*nc,opt['n_labels']], opt, name='VGGout')
		logits = tf.squeeze(logits, squeeze_dims=(1,2))
	
	return logits


def VGG(x, is_training, opt, branch=1):
	nc = opt['n_channels']
	y1 = VGG_block(x, is_training, 3, nc, 2, opt, name='1', branch=branch)
	y1 = tf.nn.max_pool(y1, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y2 = VGG_block(y1, is_training, nc, 2*nc, 2, opt, name='2', branch=branch)
	y2 = tf.nn.max_pool(y2, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y3 = VGG_block(y2, is_training, 2*nc, 4*nc, 3, opt, name='3', branch=branch)
	y3 = tf.nn.max_pool(y3, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y4 = VGG_block(y3, is_training, 4*nc, 8*nc, 3, opt, name='4', branch=branch)
	y4 = tf.nn.max_pool(y4, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y5 = VGG_block(y4, is_training, 8*nc, 8*nc, 3, opt, name='5', branch=branch)
	return [y1,y2,y3,y4,y5]


def VGG_block(x, is_training, n_in, n_channels, n_layers, opt, name='VGG_block',
				  branch=1):
	"""VGG block"""
	with tf.variable_scope('block'+name) as scope:
		l1 = conv(x, [3,3,n_in,n_channels], opt, name='1')
		l1 = bn(l1, is_training, name='1', branch=branch)
		
		l2 = conv(tf.nn.relu(l1), [3,3,n_channels,n_channels], opt, name='2')
		l2 = bn(l2, is_training, name='2', branch=branch)
		
		if n_layers == 3:
			l3 = conv(tf.nn.relu(l2), [3,3,n_channels,n_channels], opt, name='3')
		else:
			l3 = l2
		l3 = bn(l3, is_training, name='3', branch=branch)
		
		l1x1 = conv(tf.nn.relu(l3), [1,1,n_channels,n_channels], opt, name='1x1')
	return l1x1


def bn(X, train_phase, decay=0.99, name='batchNorm', branch=1):
	"""Batch normalization module.
	
	X: tf tensor
	train_phase: boolean flag True: training mode, False: test mode
	decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
	name: (default batchNorm)
	
	Source: bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
	batch-normalization-in-tensorflow"""
	n_out = X.get_shape().as_list()[3]
	
	beta = tf.get_variable('beta_'+name, dtype=tf.float32, shape=n_out,
								  initializer=tf.constant_initializer(0.0))
	gamma = tf.get_variable('gamma_'+name, dtype=tf.float32, shape=n_out,
									initializer=tf.constant_initializer(1.0))
	pop_mean = tf.get_variable('pop_mean_'+name, dtype=tf.float32,
										shape=n_out, trainable=False)
	pop_var = tf.get_variable('pop_var_'+name, dtype=tf.float32,
									  shape=n_out, trainable=False)
	batch_mean, batch_var = tf.nn.moments(X, [0,1,2], name='moments_'+name)
	
	if branch == 1:
		ema = tf.train.ExponentialMovingAverage(decay=decay)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
			pop_var_op = tf.assign(pop_var, ema.average(batch_var))
	
			with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		
		mean, var = tf.cond(train_phase, mean_var_with_update,
					lambda: (pop_mean, pop_var))
	else:
		mean, var = tf.cond(train_phase, lambda: (batch_mean, batch_var),
				lambda: (pop_mean, pop_var))
		
	normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
	return normed






































