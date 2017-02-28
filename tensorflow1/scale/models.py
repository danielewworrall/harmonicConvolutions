'''Models'''

import os
import sys
import time

import numpy as np
import tensorflow as tf

import equivariant_loss as el

from spatial_transformer import transformer


def conv(x, shape, opt, name='0', bias_init=0.01, return_params=False):
	"""Basic convolution"""
	He_initializer = tf.contrib.layers.variance_scaling_initializer()
	l2_regularizer = tf.contrib.layers.l2_regularizer(opt['weight_decay'])
	W = tf.get_variable('W'+name, shape=shape, initializer=He_initializer,
							  regularizer=l2_regularizer)
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
		y = VGG(x, opt['n_channels'], opt)
		# Transformer branch
		y1 = []
		for features in y[:-1]:
			y1.append(el.transform_features(features, t_params, f_params))
		
		# Siamese loss
		x_pre = transformer(x, t_params, (xsh[1],xsh[2]))
		scope.reuse_variables()
		y2 = VGG(x_pre, opt['n_channels'], opt)
		
		# Tail
	with tf.variable_scope('tail') as scope:
		gap = tf.reduce_mean(y[-1], axis=(1,2), keep_dims=True)
		logits = conv(gap, [1,1,8*nc,opt['n_labels']], opt, name='VGGout')
		logits = tf.squeeze(logits, squeeze_dims=(1,2))
	
	return logits, y1, y2[:-1]


def single_model(x, opt):
	"""Build siamese models for equivariance tests"""
	nc = opt['n_channels']
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('mouth') as scope:
		y = VGG(x, opt['n_channels'], opt)
				
		# Tail
	with tf.variable_scope('tail') as scope:
		gap = tf.reduce_mean(y[-1], axis=(1,2), keep_dims=True)
		logits = conv(tf.nn.relu(gap), [1,1,8*nc,opt['n_labels']], name='VGGout')
		logits = tf.squeeze(logits, squeeze_dims=(1,2))
	
	return logits


def VGG(x, nc, opt):
	y1 = VGG_block(x, 3, nc, 2, opt, name='_VGGB1')
	y1 = tf.nn.max_pool(y1, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y2 = VGG_block(y1, nc, 2*nc, 2, opt, name='_VGGB2')
	y2 = tf.nn.max_pool(y2, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y3 = VGG_block(y2, 2*nc, 4*nc, 3, opt, name='_VGGB3')
	y3 = tf.nn.max_pool(y3, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y4 = VGG_block(y3, 4*nc, 8*nc, 3, opt, name='_VGGB4')
	y4 = tf.nn.max_pool(y4, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y5 = VGG_block(y4, 8*nc, 8*nc, 3, opt, name='_VGGB5')
	return [y1,y2,y3,y4,y5]


def VGG_block(x, n_in, n_channels, n_layers, opt, name='_VGGB1'):
	"""Network in Network block"""
	with tf.variable_scope('block'+name) as scope:
		l1 = conv(x, [3,3,n_in,n_channels], opt, name='1')
		l2 = conv(tf.nn.relu(l1), [3,3,n_channels,n_channels], opt, name='2')
		if n_layers == 3:
			l2 = conv(tf.nn.relu(l2), [3,3,n_channels,n_channels], opt, name='3')
	return l2
