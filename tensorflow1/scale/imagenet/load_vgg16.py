'''Load pretrained VGG weights'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import tensorflow as tf

import equivariant_loss as el
from spatial_transformer import transformer


def main():
	fname = './vgg16_weights.npz'
	weights = np.load(fname)
	
	opt = {}
	opt['n_channels'] = 64
	opt['n_labels'] = 1000
	opt['weight_decay'] = 0.0005
	
	x = tf.placeholder(tf.float32, [32,256,256,3], name='x')
	logits = single_model(x, opt)
	
	assign_ops = []
	for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		for weight in weights:
			if weight in var.name:
				assign_ops.append(var.assign(weights[weight]))
				print('Added weights assignment op: {:s}'.format(var.name))



def conv(x, shape, opt, name='conv', bias_init=0.01):
	"""Basic convolution"""
	name = 'conv{:s}'.format(name)
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
		logits = conv(tf.nn.relu(gap), [1,1,8*nc,opt['n_labels']], opt, name='VGGout')
		logits = tf.squeeze(logits, squeeze_dims=(1,2))
	
	return logits


def VGG(x, nc, opt):
	y1 = VGG_block(x, 3, nc, 2, opt, name='1')
	y1 = tf.nn.max_pool(y1, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y2 = VGG_block(y1, nc, 2*nc, 2, opt, name='2')
	y2 = tf.nn.max_pool(y2, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y3 = VGG_block(y2, 2*nc, 4*nc, 3, opt, name='3')
	y3 = tf.nn.max_pool(y3, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y4 = VGG_block(y3, 4*nc, 8*nc, 3, opt, name='4')
	y4 = tf.nn.max_pool(y4, (1,2,2,1), (1,2,2,1), padding='VALID')
	
	y5 = VGG_block(y4, 8*nc, 8*nc, 3, opt, name='5')
	return [y1,y2,y3,y4,y5]


def VGG_block(x, n_in, n_channels, n_layers, opt, name='VGG_block'):
	"""Network in Network block"""
	with tf.variable_scope('block'+name) as scope:
		l1 = conv(x, [3,3,n_in,n_channels], opt, name='{:s}_1'.format(name))
		l2 = conv(tf.nn.relu(l1), [3,3,n_channels,n_channels], opt, name='{:s}_2'.format(name))
		if n_layers == 3:
			l2 = conv(tf.nn.relu(l2), [3,3,n_channels,n_channels], opt, name='{:s}_3'.format(name))
	return l2



if __name__ == '__main__':
	main()