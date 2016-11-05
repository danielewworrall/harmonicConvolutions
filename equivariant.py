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
def deep_complex_bias(opt, x, bs, phase_train, device='/cpu:0'):
	"""The deep_complex_bias architecture. Current test time score on rot-MNIST
	is 97.81%---state-of-the-art
	"""
	# Sure layers weight & bias
	order = 3
	nf = opt['n_filters']
	nf2 = int(nf*opt['filter_gain'])
	nf3 = int(nf*(opt['filter_gain']**2.))
	bs = opt['batch_size']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1' : get_weights_dict([[6,],[5,],[5,]], opt['n_channels'], nf, std_mult=sm, name='W1', device=device),
			'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=sm, name='W2', device=device),
			'w3' : get_weights_dict([[6,],[5,],[5,]], nf, nf2, std_mult=sm, name='W3', device=device),
			'w4' : get_weights_dict([[6,],[5,],[5,]], nf2, nf2, std_mult=sm, name='W4', device=device),
			'w5' : get_weights_dict([[6,],[5,],[5,]], nf2, nf3, std_mult=sm, name='W5', device=device),
			'w6' : get_weights_dict([[6,],[5,],[5,]], nf3, nf3, std_mult=sm, name='W6', device=device),
			'w7' : get_weights_dict([[6,],[5,],[5,]], nf3, opt['n_classes'], std_mult=sm, name='W7', device=device),
		}
		
		biases = {
			'b1' : get_bias_dict(nf, 2, name='b1', device=device),
			'b2' : get_bias_dict(nf, 2, name='b2', device=device),
			'b3' : get_bias_dict(nf2, 2, name='b3', device=device),
			'b4' : get_bias_dict(nf2, 2, name='b4', device=device),
			'b5' : get_bias_dict(nf3, 2, name='b5', device=device),
			'b6' : get_bias_dict(nf3, 2, name='b6', device=device),
			'b7' : tf.get_variable('b7', dtype=tf.float32, shape=[opt['n_classes']],
				initializer=tf.constant_initializer(1e-2)),
			'psi1' : get_phase_dict(1, nf, 2, name='psi1', device=device),
			'psi2' : get_phase_dict(nf, nf, 2, name='psi2', device=device),
			'psi3' : get_phase_dict(nf, nf2, 2, name='psi3', device=device),
			'psi4' : get_phase_dict(nf2, nf2, 2, name='psi4', device=device),
			'psi5' : get_phase_dict(nf2, nf3, 2, name='psi5', device=device),
			'psi6' : get_phase_dict(nf3, nf3, 2, name='psi6', device=device)
		}
		# Reshape input picture -- square inputs for now
		size = opt['dim'] - 2*opt['crop_shape']
		x = tf.reshape(x, shape=[bs,size,size,opt['n_channels']])
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train,
								 name='batchNorm1', device=device)
	
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
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train,
								 name='batchNorm2', device=device)
	
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
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train,
								 name='batchNorm3', device=device)

	# LAYER 7
	with tf.name_scope('block4') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=5,
								 padding='SAME', name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7'])

def deep_stable(opt, x, bs, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = opt['n_filters']
	nf2 = int(nf*opt['filter_gain'])
	nf3 = int(nf*(opt['filter_gain']**2.))
	bs = opt['batch_size']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1' : get_weights_dict([[6,],[5,]], opt['n_channels'], nf, std_mult=sm, name='W1', device=device),
			'w2' : get_weights_dict([[6,],[5,]], nf, nf, std_mult=sm, name='W2', device=device),
			'w3' : get_weights_dict([[6,],[5,]], nf, nf2, std_mult=sm, name='W3', device=device),
			'w4' : get_weights_dict([[6,],[5,]], nf2, nf2, std_mult=sm, name='W4', device=device),
			'w5' : get_weights_dict([[6,],[5,]], nf2, nf3, std_mult=sm, name='W5', device=device),
			'w6' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W6', device=device),
			'w7' : get_weights_dict([[6,],[5,]], nf3, opt['n_classes'], std_mult=sm, name='W7', device=device),
		}
		
		biases = {
			'b1' : get_bias_dict(nf, order, name='b1', device=device),
			'b2' : get_bias_dict(nf, order, name='b2', device=device),
			'b3' : get_bias_dict(nf2, order, name='b3', device=device),
			'b4' : get_bias_dict(nf2, order, name='b4', device=device),
			'b5' : get_bias_dict(nf3, order, name='b5', device=device),
			'b6' : get_bias_dict(nf3, order, name='b6', device=device),
			'b7' : tf.get_variable('b7', dtype=tf.float32, shape=[opt['n_classes']],
				initializer=tf.constant_initializer(1e-2)),
			'psi1' : get_phase_dict(1, nf, order, name='psi1', device=device),
			'psi2' : get_phase_dict(nf, nf, order, name='psi2', device=device),
			'psi3' : get_phase_dict(nf, nf2, order, name='psi3', device=device),
			'psi4' : get_phase_dict(nf2, nf2, order, name='psi4', device=device),
			'psi5' : get_phase_dict(nf2, nf3, order, name='psi5', device=device),
			'psi6' : get_phase_dict(nf3, nf3, order, name='psi6', device=device)
		}
		# Reshape input picture -- square inputs for now
		size = opt['dim'] - 2*opt['crop_shape']
		x = tf.reshape(x, shape=[bs,size,size,opt['n_channels']])
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train,
								 name='batchNorm1', device=device)
	
	with tf.name_scope('block2') as scope:
		cv2 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)

		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train,
								 name='batchNorm2', device=device)
	
	with tf.name_scope('block3') as scope:
		cv4 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 5
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)

		# LAYER 6
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train,
								 name='batchNorm3', device=device)

	# LAYER 7
	with tf.name_scope('block4') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=5,
								 padding='SAME', name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7'])

def deep_plankton(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = opt['n_filters']
	nf2 = int(nf*opt['filter_gain'])
	nf3 = int(nf*(opt['filter_gain']**2.))
	bs = opt['batch_size']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1' : get_weights_dict([[6,],[5,],[5,]], opt['n_channels'], nf, std_mult=sm, name='W1', device=device),
			'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=sm, name='W2', device=device),
			'w3' : get_weights_dict([[6,],[5,]], nf, nf2, std_mult=sm, name='W3', device=device),
			'w4' : get_weights_dict([[6,],[5,]], nf2, nf2, std_mult=sm, name='W4', device=device),
			'w5' : get_weights_dict([[6,],[5,]], nf2, nf3, std_mult=sm, name='W5', device=device),
			'w6' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W6', device=device),
			'w7' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W7', device=device),
			'w8' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W8', device=device),
			'w9' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W9', device=device),
			'w10' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W10', device=device),
			'w11' : get_weights_dict([[6,],[5,]], nf3, opt['n_classes'], std_mult=sm, name='W11', device=device),
		}
		
		biases = {
			'b1' : get_bias_dict(nf, order+1, name='b1', device=device),
			'b2' : get_bias_dict(nf, order+1, name='b2', device=device),
			'b3' : get_bias_dict(nf2, order, name='b3', device=device),
			'b4' : get_bias_dict(nf2, order, name='b4', device=device),
			'b5' : get_bias_dict(nf3, order, name='b5', device=device),
			'b6' : get_bias_dict(nf3, order, name='b6', device=device),
			'b7' : get_bias_dict(nf3, order, name='b7', device=device),
			'b8' : get_bias_dict(nf3, order, name='b8', device=device),
			'b9' : get_bias_dict(nf3, order, name='b9', device=device),
			'b10' : get_bias_dict(nf3, order, name='b10', device=device),
			'b11' : tf.get_variable('b11', dtype=tf.float32, shape=[opt['n_classes']],
				initializer=tf.constant_initializer(1e-2)),
			'psi1' : get_phase_dict(1, nf, order+1, name='psi1', device=device),
			'psi2' : get_phase_dict(nf, nf, order+1, name='psi2', device=device),
			'psi3' : get_phase_dict(nf, nf2, order, name='psi3', device=device),
			'psi4' : get_phase_dict(nf2, nf2, order, name='psi4', device=device),
			'psi5' : get_phase_dict(nf2, nf3, order, name='psi5', device=device),
			'psi6' : get_phase_dict(nf3, nf3, order, name='psi6', device=device),
			'psi7' : get_phase_dict(nf3, nf3, order, name='psi7', device=device),
			'psi8' : get_phase_dict(nf3, nf3, order, name='psi8', device=device),
			'psi9' : get_phase_dict(nf3, nf3, order, name='psi9', device=device),
			'psi10' : get_phase_dict(nf3, nf3, order, name='psi10', device=device)
		}
		# Reshape input picture -- square inputs for now
		size = opt['dim'] - 2*opt['crop_shape']
		x = tf.reshape(x, shape=[bs,size,size,opt['n_channels']])
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train,
								 name='batchNorm1', device=device)
	
	with tf.name_scope('block2') as scope:
		cv3 = mean_pooling(cv2, ksize=(1,3,3,1), strides=(1,2,2,1))
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv3, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train,
								 name='batchNorm2', device=device)
	
	with tf.name_scope('block3') as scope:
		cv5 = mean_pooling(cv4, ksize=(1,3,3,1), strides=(1,2,2,1))
		cv5 = complex_input_rotated_conv(cv5, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='6')
		cv6 = complex_nonlinearity(cv6, biases['b6'], tf.nn.relu)
		cv7 = complex_input_rotated_conv(cv6, weights['w7'], biases['psi7'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='7')
		cv7 = complex_batch_norm(cv7, tf.nn.relu, phase_train,
								 name='batchNorm3', device=device)
	
	with tf.name_scope('block4') as scope:
		cv8 = mean_pooling(cv7, ksize=(1,3,3,1), strides=(1,2,2,1))
		cv8 = complex_input_rotated_conv(cv8, weights['w8'], biases['psi8'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='8')
		cv8 = complex_nonlinearity(cv8, biases['b8'], tf.nn.relu)
		cv9 = complex_input_rotated_conv(cv8, weights['w9'], biases['psi9'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='9')
		cv9 = complex_nonlinearity(cv9, biases['b9'], tf.nn.relu)
		cv10 = complex_input_rotated_conv(cv9, weights['w10'], biases['psi10'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='10')
		cv10 = complex_batch_norm(cv10, tf.nn.relu, phase_train,
								 name='batchNorm4', device=device)
	
	with tf.name_scope('block5') as scope:
		cv11 = complex_input_conv(cv10, weights['w11'], filter_size=5,
								 padding='SAME', name='11')
		cv11 = tf.reduce_mean(sum_magnitudes(cv11), reduction_indices=[1,2])
		return tf.nn.bias_add(cv11, biases['b11'])


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


##### CUSTOM FUNCTIONS FOR MAIN SCRIPT #####
def minibatcher(inputs, targets, batch_size, shuffle=False, augment=False,
				img_shape=(95,95), crop_shape=10):
	"""Input and target are minibatched. Returns a generator"""
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = np.arange(batch_size) + start_idx
		# Data augmentation
		im = []
		for i in xrange(len(excerpt)):
			img = inputs[excerpt[i]]
			if augment:
				# We use shuffle as a proxy for training
				if shuffle:
					img = preprocess(img, img_shape, crop_shape)
				else:
					img = np.reshape(img, img_shape)
					img = img[crop_shape:-crop_shape,crop_shape:-crop_shape]
					img = np.reshape(img, [1,np.prod(img.shape)])
			im.append(img)
		im = np.vstack(im)
		yield im, targets[excerpt]

def preprocess(im, im_shape, crop_shape):
	'''Data normalizations and augmentations'''
	# Random rotations
	im = np.reshape(im, im_shape)
	angle = np.random.rand() * 360.
	im = sciint.rotate(im, angle, reshape=False)
	# Random crops
	crops = np.random.randint(2*crop_shape, size=2)
	crop_im_shape = (im_shape[0] - 2*crop_shape, im_shape[1] - 2*crop_shape)
	im = im[crops[0]:crops[0]+crop_im_shape[0],crops[1]:crops[1]+crop_im_shape[1]]
	# Random fliplr
	if np.random.rand() > 0.5:
		im = im[:,::-1]
	return np.reshape(im, [1,np.prod(im.shape)])

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

def get_learning_rate(opt, current, best, counter, learning_rate):
	"""If have not seen accuracy improvement in delay epochs, then divide 
	learning rate by 10
	"""
	if current > best:
		best = current
		counter = 0
	elif counter > opt['delay']:
		learning_rate = learning_rate / opt['lr_div']
		counter = 0
	else:
		counter += 1
	return (best, counter, learning_rate)