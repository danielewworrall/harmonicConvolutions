'''Equivariant tests'''

import os
import sys
import time

import numpy as np
import tensorflow as tf
import scipy as sp

from harmonic_network_ops import get_weights

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


def get_weights_dict_(shape, max_order, std_mult=0.4, name='W', device='/cpu:0'):
	"""Return a dict of weights.
	
	shape: list of filter shape [h,w,i,o] --- note we use h=w
	max_order: returns weights for m=0,1,...,max_order
	std_mult: He init scaled by std_mult (default 0.4)
	name: (default 'W')
	dev: (default /cpu:0)
	"""
	K = shape[2]
	M = max_order+1
	weights_dict = {}
	radius = (shape[0]+1)/2
	n_rings = (radius*(radius+1))/2
	phi = get_nitems_per_ring(shape[0])
	for i in xrange(max_order+1):
		sh = [n_rings-(i>0)] + shape[2:]
		nm = name + '_' + str(i)
		if i == 0:
			phi = np.asarray(phi)
			phi[0] = 1./M
			stddev = np.sqrt(1./(K*M*n_rings*phi))
			W_init = stddev[:,np.newaxis,np.newaxis]*np.random.standard_normal(sh).astype(np.float32)
		else:
			stddev = np.sqrt(1./(K*M*(n_rings-1)*np.asarray(phi)[1:]))
			W_init = stddev[:,np.newaxis,np.newaxis]**np.random.standard_normal(sh)
		W_init = tf.constant(W_init, tf.float32)
		weights_dict[i] = get_weights(None, W_init=W_init, name=nm, device=device)
	return weights_dict


def get_nitems_per_ring(ksize):
	"""Return the number of elements per ring"""
	radius = (ksize+1)/2
	n_rings = (radius*(radius+1))/2
	lin = (np.arange(ksize) - (ksize-1)/2)[:,np.newaxis]
	ones = np.ones((1,ksize))
	X = np.dot(lin, ones)
	Y = -X.T
	R2 = X**2 + Y**2
	__, c = np.unique(R2, return_counts=True)
	return c


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