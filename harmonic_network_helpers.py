'''Equivariant tests'''

import os
import sys
import time

import numpy as np
import tensorflow as tf
import scipy as sp

#----------CORE FUNCTIONS FOR LAYER CREATION---------- 
def conv2d(X, V, b=None, strides=(1,1,1,1), padding='VALID', name='conv2d'):
	"""conv2d wrapper. Supply input X, weights V and optional bias"""
	VX = tf.nn.conv2d(X, V, strides=strides, padding=padding, name=name+'_')
	if b is not None:
		VX = tf.nn.bias_add(VX, b)
	return VX

def maxpool2d(X, k=2):
	"""Tied max pool. k is the stride and pool size"""
	return tf.nn.max_pool(X, ksize=[1,k,k,1], strides=[1,k,k,1],
						  padding='VALID')

def get_weights_dict(comp_shape, in_shape, out_shape, std_mult=0.4, name='W',
					 device='/cpu:0'):
	"""Return a dict of weights for use with real_input_equi_conv. comp_shape is
	a list of the number of elements per Fourier base. For 3x3 weights use
	[3,2,2,2]. I currently assume order increasing from 0.
	"""
	weights_dict = {}
	for i, cs in enumerate(comp_shape):
		shape = cs + [in_shape,out_shape]
		weights_dict[i] = get_weights(shape, std_mult=std_mult,
									  name=name+'_'+str(i), device=device)
	return weights_dict

def get_bias_dict(n_filters, order, name='b', device='/cpu:0'):
	"""Return a dict of biases"""
	with tf.device(device):
		bias_dict = {}
		for i in xrange(order+1):
			bias = tf.get_variable(name+'_'+str(i), dtype=tf.float32,
								   shape=[n_filters],
				initializer=tf.constant_initializer(1e-2))
			bias_dict[i] = bias
	return bias_dict

def get_phase_dict(n_in, n_out, order, name='b',device='/cpu:0'):
	"""Return a dict of phase offsets"""
	with tf.device(device):
		phase_dict = {}
		for i in xrange(order+1):
			init = np.random.rand(1,1,n_in,n_out) * 2. *np.pi
			init = np.float32(init)
			phase = tf.get_variable(name+'_'+str(i), dtype=tf.float32,
									shape=[1,1,n_in,n_out],
				initializer=tf.constant_initializer(init))
			phase_dict[i] = phase
	return phase_dict

#----------ADDITIONAL FUNCTIONS FOR CREATING BLOCKS----------
def up_block(x, d, w1, w2, p1, p2, b, pt, name, device):
	'''Upsampling block'''
	with tf.name_scope(name) as scope:
		x = tf.image.resize_bilinear(x, size, align_corners=True)
		x = tf.concat(3, [x,d])
		cv1 = complex_input_rotated_conv(x, w1, p1, filter_size=3,
											 output_orders=[0,1],
											 padding='SAME', name=name+'_1')
		cv1 = complex_nonlinearity(cv1, b, tf.nn.relu)
	
		cv2 = complex_input_rotated_conv(cv1, w2, p2, filter_size=3,
										 output_orders=[0,1], padding='SAME',
										 name=name+'_2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, pt, name=name+'_bn',
								 device=device)
		return out

def down_block(in_, x, w1, w2, p1, p2, b, pt, name, device):
	'''Downsampling block'''
	with tf.name_scope(name) as scope:
		if in_:
			cv1 = real_input_rotated_conv(x, w1, p1, filter_size=3,
										  padding='SAME', name=name+'_1')
		else:
			cv1 = complex_input_rotated_conv(x, w1, p1, filter_size=3,
											 output_orders=[0,1],
											 padding='SAME', name=name+'_1')
		cv1 = complex_nonlinearity(cv1, b, tf.nn.relu)
	
		cv2 = complex_input_rotated_conv(cv1, w2, p2, filter_size=3,
										 output_orders=[0,1], padding='SAME',
										 name=name+'_2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, pt, name=name+'_bn',
								 device=device)
		out = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		return out, cv2