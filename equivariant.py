'''Equivariant tests'''

import os
import sys
import time

#import cv2
import numpy as np
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import tensorflow as tf

import input_data

from steer_conv import *

from matplotlib import pyplot as plt

import scipy as sp
from scipy import ndimage
from scipy import misc

##### HELPERS #####
def checkFolder(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


##### MODELS #####
def deep_complex_bias(x, drop_prob, n_filters, n_rows, n_cols, n_channels, size_after_conv, n_classes,
 bs, phase_train, std_mult, filter_gain=2.0, device='/cpu:0'):
	"""The deep_complex_bias architecture. Current test time score is 94.7% for 7 layers 
	deep, 5 filters
	"""
	# Sure layers weight & bias
	order = 3
	nf = n_filters
	nf2 = int(n_filters*filter_gain)
	nf3 = int(n_filters*(filter_gain**2.))
	with tf.device(device):
		weights = {
			'w1' : get_weights_dict([[6,],[5,],[5,]], n_channels, nf, std_mult=std_mult, name='W1', device=device),
			'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W2', device=device),
			'w3' : get_weights_dict([[6,],[5,],[5,]], nf, nf2, std_mult=std_mult, name='W3', device=device),
			'w4' : get_weights_dict([[6,],[5,],[5,]], nf2, nf2, std_mult=std_mult, name='W4', device=device),
			'w5' : get_weights_dict([[6,],[5,],[5,]], nf2, nf3, std_mult=std_mult, name='W5', device=device),
			'w6' : get_weights_dict([[6,],[5,],[5,]], nf3, nf3, std_mult=std_mult, name='W6', device=device),
			'w7' : get_weights_dict([[6,],[5,],[5,]], nf3, n_classes, std_mult=std_mult, name='W7', device=device),
		}
		
		biases = {
			'b1' : get_bias_dict(nf, 2, name='b1', device=device),
			'b2' : get_bias_dict(nf, 2, name='b2', device=device),
			'b3' : get_bias_dict(nf2, 2, name='b3', device=device),
			'b4' : get_bias_dict(nf2, 2, name='b4', device=device),
			'b5' : get_bias_dict(nf3, 2, name='b5', device=device),
			'b6' : get_bias_dict(nf3, 2, name='b6', device=device),
			'b7' : tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='b7'),
			'psi1' : get_phase_dict(1, nf, 2, name='psi1', device=device),
			'psi2' : get_phase_dict(nf, nf, 2, name='psi2', device=device),
			'psi3' : get_phase_dict(nf, nf2, 2, name='psi3', device=device),
			'psi4' : get_phase_dict(nf2, nf2, 2, name='psi4', device=device),
			'psi5' : get_phase_dict(nf2, nf3, 2, name='psi5', device=device),
			'psi6' : get_phase_dict(nf3, nf3, 2, name='psi6', device=device)
		}
		# Reshape input picture
		x = tf.reshape(x, shape=[bs, n_rows, n_cols, n_channels])
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train, name='batchNorm1', device=device)
	
	with tf.name_scope('block2') as scope:
		cv2 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)

		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train, name='batchNorm2', device=device)
	
	with tf.name_scope('block3') as scope:
		cv4 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 5
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)

		# LAYER 6
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train, name='batchNorm3', device=device)

	# LAYER 7
	with tf.name_scope('block4') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=5,
								 padding='SAME', name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7'])

##### CUSTOM BLOCKS FOR MODEL #####
def res_block(x, w1, w2, psi1, psi2, b, phase_train, filter_size=5,
			  strides=(1,2,2,1), name='1'):
	"""Residual block"""
		
	with tf.name_scope('block'+name) as scope:
		cv1 = complex_input_rotated_conv(x, w1, psi1, filter_size=filter_size,
									  output_orders=[0,1,2], padding='SAME',
									  strides=strides, name='1')
		cv1 = complex_nonlinearity(cv1, b, tf.nn.relu)
		
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, w2, psi2, filter_size=filter_size,
										 output_orders=[0,1,2], padding='SAME',
										 name='2')
		cv2 = complex_batch_norm(cv2, lambda x:x, phase_train, device=device)
		
		# Shortcut across equal rotation order complex feature maps
		for order, val in x.iteritems():
			s0 = tf.nn.avg_pool(val[0], (1,strides[1],strides[2],1), strides,
								padding='VALID', name='s'+str(order)+'_0')
			p = tf.maximum(cv2[order][0].get_shape()[3]-s0.get_shape()[3],0)
			s0 = tf.pad(s0,[[0,0],[0,0],[0,0],[0,p]])
			
			s1 = tf.nn.avg_pool(val[0], (1,strides[1],strides[2],1), strides,
								padding='VALID', name='s'+str(order)+'_1')
			s1 = tf.pad(s1,[[0,0],[0,0],[0,0],[0,p]])
			
			cv2[order] = (cv2[order][0]+s0, cv2[order][1]+s1)
			
		return cv2
		

def conv2d(X, V, b=None, strides=(1,1,1,1), padding='VALID', name='conv2d'):
	"""conv2d wrapper. Supply input X, weights V and optional bias"""
	VX = tf.nn.conv2d(X, V, strides=strides, padding=padding, name=name+'_')
	if b is not None:
		VX = tf.nn.bias_add(VX, b)
	return VX

def maxpool2d(X, k=2):
	"""Tied max pool. k is the stride and pool size"""
	return tf.nn.max_pool(X, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def get_weights_dict(comp_shape, in_shape, out_shape, std_mult=0.4, name='W', device='/cpu:0'):
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
			bias = tf.get_variable(name+'_'+str(i), dtype=tf.float32, shape=[n_filters],
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
			phase = tf.get_variable(name+'_'+str(i), dtype=tf.float32, shape=[1,1,n_in,n_out],
				initializer=tf.constant_initializer(init))
			phase_dict[i] = phase
	return phase_dict


##### CUSTOM FUNCTIONS FOR MAIN SCRIPT #####
def minibatcher(inputs, targets, batch_size, shuffle=False):
	"""Input and target are minibatched. Returns a generator"""
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield inputs[excerpt], targets[excerpt]

def save_model(saver, saveDir, sess):
	"""Save a model checkpoint"""
	save_path = saver.save(sess, saveDir + "checkpoints/model.ckpt")
	print("Model saved in file: %s" % save_path)

def restore_model(saver, saveDir, sess):
	"""Save a model checkpoint"""
	save_path = saver.restore(sess, saveDir + "checkpoints/model.ckpt")
	print("Model restored from file: %s" % save_path)

def rotate_feature_maps(X, n_angles):
	"""Rotate feature maps"""
	X = np.reshape(X, [28,28])
	X_ = []
	for angle in np.linspace(0, 360, num=n_angles):
		X_.append(sciint.rotate(X, angle, reshape=False))
	X_ = np.stack(X_, axis=0)
	X_ = np.reshape(X_, [-1,784])
	return X_

def get_learning_rate(current, best, counter, learning_rate, delay=15):
	"""If have not seen accuracy improvement in delay epochs, then divide 
	learning rate by 10
	"""
	if current > best:
		best = current
		counter = 0
	elif counter > delay:
		learning_rate = learning_rate / 10.
		counter = 0
	else:
		counter += 1
	return (best, counter, learning_rate)