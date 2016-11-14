'''Equivariant tests'''

import os
import sys
import time

#import cv2
import numpy as np
import scipy.linalg as scilin
from scipy.ndimage import distance_transform_edt
import scipy.ndimage.interpolation as sciint
import skimage.color as skco
import skimage.exposure as skiex
import skimage.io as skio
import skimage.morphology as skmo
import skimage.transform as sktr
import tensorflow as tf

import input_data

from steer_conv import *

from matplotlib import pyplot as plt

import scipy as sp
from scipy import ndimage

##### HELPERS #####
def checkFolder(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


##### MODELS #####
def deep_stable(opt, x, phase_train, device='/cpu:0'):
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
	
	fms = []
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		fms.append(cv1)	
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train,
								 name='batchNorm1', device=device)
		fms.append(cv2)
	with tf.name_scope('block2') as scope:
		cv2 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)
		fms.append(cv3)
		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train,
								 name='batchNorm2', device=device)
		fms.append(cv4)
	with tf.name_scope('block3') as scope:
		cv4 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 5
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
		fms.append(cv5)
		# LAYER 6
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train,
								 name='batchNorm3', device=device)
		fms.append(cv6)
	# LAYER 7
	with tf.name_scope('block4') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=5,
								 padding='SAME', name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7']), fms

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
		cv1 = complex_batch_norm(cv1, tf.nn.relu, phase_train, name='bn1', device=device)
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train, name='bn2', device=device)
	
	with tf.name_scope('block2') as scope:
		cv3 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv3, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='3')
		cv3 = complex_batch_norm(cv3, tf.nn.relu, phase_train, name='bn3', device=device)
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train, name='b4', device=device)
	
	with tf.name_scope('block3') as scope:
		cv5 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv5 = complex_input_rotated_conv(cv5, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='5')
		cv5 = complex_batch_norm(cv5, tf.nn.relu, phase_train, name='bn5', device=device)
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='6')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train, name='bn6', device=device)
		cv7 = complex_input_rotated_conv(cv6, weights['w7'], biases['psi7'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='7')
		cv7 = complex_batch_norm(cv7, tf.nn.relu, phase_train, name='bn7', device=device)
	
	with tf.name_scope('block4') as scope:
		cv8 = mean_pooling(cv7, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv8 = complex_input_rotated_conv(cv8, weights['w8'], biases['psi8'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='8')
		cv8 = complex_batch_norm(cv8, tf.nn.relu, phase_train, name='bn8', device=device)
		cv9 = complex_input_rotated_conv(cv8, weights['w9'], biases['psi9'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='9')
		cv9 = complex_batch_norm(cv9, tf.nn.relu, phase_train, name='bn9', device=device)
		cv10 = complex_input_rotated_conv(cv9, weights['w10'], biases['psi10'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='10')
		cv10 = complex_batch_norm(cv10, tf.nn.relu, phase_train, name='bn10', device=device)
	
	with tf.name_scope('block5') as scope:
		cv11 = complex_input_conv(cv10, weights['w11'], filter_size=5,
								 padding='SAME', name='11')
		cv11 = tf.reduce_mean(sum_magnitudes(cv11), reduction_indices=[1,2])
		return tf.nn.bias_add(cv11, biases['b11'])

def deep_cifar(opt, x, phase_train, device='/cpu:0'):
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
			'w3' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=sm, name='W3', device=device),
			'w4' : get_weights_dict([[6,],[5,]], nf, nf2, std_mult=sm, name='W4', device=device),
			'w5' : get_weights_dict([[6,],[5,]], nf2, nf2, std_mult=sm, name='W5', device=device),
			'w6' : get_weights_dict([[6,],[5,]], nf2, nf2, std_mult=sm, name='W6', device=device),
			'w7' : get_weights_dict([[6,],[5,]], nf2, nf3, std_mult=sm, name='W7', device=device),
			'w8' : get_weights_dict([[3,],[2,]], nf3, nf3, std_mult=sm, name='W8', device=device),
			'w9' : get_weights_dict([[3,],[2,]], nf3, opt['n_classes'], std_mult=sm, name='W9', device=device),
		}
		
		biases = {
			'b1' : get_bias_dict(nf, order+1, name='b1', device=device),
			'b2' : get_bias_dict(nf, order+1, name='b2', device=device),
			'b3' : get_bias_dict(nf, order, name='b3', device=device),
			'b4' : get_bias_dict(nf2, order, name='b4', device=device),
			'b5' : get_bias_dict(nf2, order, name='b5', device=device),
			'b6' : get_bias_dict(nf2, order, name='b6', device=device),
			'b7' : get_bias_dict(nf3, order, name='b7', device=device),
			'b8' : get_bias_dict(nf3, order, name='b8', device=device),
			'b9' : tf.get_variable('b9', dtype=tf.float32, shape=[opt['n_classes']],
				initializer=tf.constant_initializer(1e-2)),
			'psi1' : get_phase_dict(1, nf, order+1, name='psi1', device=device),
			'psi2' : get_phase_dict(nf, nf, order+1, name='psi2', device=device),
			'psi3' : get_phase_dict(nf, nf, order+1, name='psi3', device=device),
			'psi4' : get_phase_dict(nf, nf2, order, name='psi4', device=device),
			'psi5' : get_phase_dict(nf2, nf2, order, name='psi5', device=device),
			'psi6' : get_phase_dict(nf2, nf2, order, name='psi6', device=device),
			'psi7' : get_phase_dict(nf2, nf3, order, name='psi7', device=device),
			'psi8' : get_phase_dict(nf3, nf3, order, name='psi8', device=device),
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
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train, name='bn1', device=device)	
		cv3 = mean_pooling(cv2, ksize=(1,3,3,1), strides=(1,2,2,1))
		cv3 = complex_input_rotated_conv(cv3, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)

	with tf.name_scope('block2') as scope:
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train, name='bn2', device=device)
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
		cv5 = mean_pooling(cv5, ksize=(1,3,3,1), strides=(1,2,2,1))
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='6')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train, name='bn3', device=device)

	with tf.name_scope('block3') as scope:
		cv7 = complex_input_rotated_conv(cv6, weights['w7'], biases['psi7'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='7')
		cv7 = complex_nonlinearity(cv7, biases['b7'], tf.nn.relu)
		cv8 = complex_input_rotated_conv(cv7, weights['w8'], biases['psi8'],
										 filter_size=3, output_orders=[0,1],
										 padding='SAME', name='8')
		cv8 = complex_batch_norm(cv8, tf.nn.relu, phase_train, name='bn4', device=device)
		cv9 = complex_input_conv(cv8, weights['w9'], filter_size=3, output_orders=[0,1],
										 padding='SAME', name='9')
		cv9 = tf.reduce_mean(sum_magnitudes(cv9), reduction_indices=[1,2])
		return tf.nn.bias_add(cv9, biases['b9'])

def deep_bsd(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = int(opt['n_filters'])
	nf2 = int((opt['filter_gain'])*nf)
	nf3 = int((opt['filter_gain']**2)*nf)
	nf4 = int((opt['filter_gain']**3)*nf)
	bs = opt['batch_size']
	size = opt['dim']
	size2 = opt['dim2']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1_1' : get_weights_dict([[3,],[2,]], opt['n_channels'], nf, std_mult=sm, name='W1_1', device=device),
			'w1_2' : get_weights_dict([[3,],[2,]], nf, nf, std_mult=sm, name='W1_2', device=device),
			'w2_1' : get_weights_dict([[3,],[2,],], nf, nf2, std_mult=sm, name='W2_1', device=device),
			'w2_2' : get_weights_dict([[3,],[2,],], nf2, nf2, std_mult=sm, name='W2_2', device=device),
			'w3_1' : get_weights_dict([[3,],[2,],], nf2, nf3, std_mult=sm, name='W3_1', device=device),
			'w3_2' : get_weights_dict([[3,],[2,],], nf3, nf3, std_mult=sm, name='W3_2', device=device),
			'w4_1' : get_weights_dict([[3,],[2,],], nf3, nf4, std_mult=sm, name='W4_1', device=device),
			'w4_2' : get_weights_dict([[3,],[2,],], nf4, nf4, std_mult=sm, name='W4_2', device=device),
			'w5_1' : get_weights_dict([[3,],[2,],], nf4, nf4, std_mult=sm, name='W5_1', device=device),
			'w5_2' : get_weights_dict([[3,],[2,],], nf4, nf4, std_mult=sm, name='W5_2', device=device),
			#'w_out' : tf.get_variable('w_out', dtype=tf.float32, shape=[1,1,(order+1)*nf4,opt['n_classes']],
			#						  initializer=tf.constant_initializer(1e-2))
		}
		
		biases = {
			'b1_2' : tf.get_variable('b1_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			'b2_2' : tf.get_variable('b2_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			'b3_2' : tf.get_variable('b3_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			'b4_2' : tf.get_variable('b4_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			'b5_2' : tf.get_variable('b5_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			#'b6' : tf.get_variable('b6', dtype=tf.float32, shape=[opt['n_classes']],
			#					   initializer=tf.constant_initializer(1e-2)),
			'cb1_1' : get_bias_dict(nf, order, name='cb1_1', device=device),
			'cb2_1' : get_bias_dict(nf2, order, name='cb2_1', device=device),
			'cb3_1' : get_bias_dict(nf3, order, name='cb3_1', device=device),
			'cb4_1' : get_bias_dict(nf4, order, name='cb4_1', device=device),
			'cb5_1' : get_bias_dict(nf4, order, name='cb5_1', device=device),
			'fuse' : tf.get_variable('fuse', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2))
		}
		
		side_weights = {
			'sw1' : tf.get_variable('sw1', dtype=tf.float32, shape=[1,1,(order+1)*nf,1],
				initializer=tf.constant_initializer(1e-2)),
			'sw2' : tf.get_variable('sw2', dtype=tf.float32, shape=[1,1,(order+1)*nf2,1],
				initializer=tf.constant_initializer(1e-2)),
			'sw3' : tf.get_variable('sw3', dtype=tf.float32, shape=[1,1,(order+1)*nf3,1],
				initializer=tf.constant_initializer(1e-2)),
			'sw4' : tf.get_variable('sw4', dtype=tf.float32, shape=[1,1,(order+1)*nf4,1],
				initializer=tf.constant_initializer(1e-2)),
			'sw5' : tf.get_variable('sw5', dtype=tf.float32, shape=[1,1,(order+1)*nf4,1],
				initializer=tf.constant_initializer(1e-2)),
			'h1' : tf.get_variable('h1', dtype=tf.float32, shape=[1,1,5,1],
				initializer=tf.constant_initializer(1e-2))
		}
		
		psis = {
			'psi1_1' : get_phase_dict(1, nf, order, name='psi1_1', device=device),
			'psi1_2' : get_phase_dict(nf, nf, order, name='psi1_2', device=device),
			'psi2_1' : get_phase_dict(nf, nf2, order, name='psi2_1', device=device),
			'psi2_2' : get_phase_dict(nf2, nf2, order, name='psi2_2', device=device),
			'psi3_1' : get_phase_dict(nf2, nf3, order, name='psi3_1', device=device),
			'psi3_2' : get_phase_dict(nf3, nf3, order, name='psi3_2', device=device),
			'psi4_1' : get_phase_dict(nf3, nf4, order, name='psi4_1', device=device),
			'psi4_2' : get_phase_dict(nf4, nf4, order, name='psi4_2', device=device),
			'psi5_1' : get_phase_dict(nf4, nf4, order, name='psi5_1', device=device),
			'psi5_2' : get_phase_dict(nf4, nf4, order, name='psi5_2', device=device),
		}
		
		x = tf.reshape(x, tf.pack([opt['batch_size'],size,size2,3]))
		fm = {}
		
	# Convolutional Layers
	with tf.name_scope('stage1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1_1'], psis['psi1_1'],
				 filter_size=3, padding='SAME', name='1_1')
		cv1 = complex_nonlinearity(cv1, biases['cb1_1'], tf.nn.relu)
	
		cv2 = complex_input_rotated_conv(cv1, weights['w1_2'], psis['psi1_2'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='1_2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train, name='bn1', device=device)
		fm[1] = conv2d(stack_magnitudes(cv2), side_weights['sw1']) #, b=biases['b1_2'])
	
	with tf.name_scope('stage2') as scope:
		cv3 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv3 = complex_input_rotated_conv(cv2, weights['w2_1'], psis['psi2_1'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='2_1')
		cv3 = complex_nonlinearity(cv3, biases['cb2_1'], tf.nn.relu)
	
		cv4 = complex_input_rotated_conv(cv3, weights['w2_2'], psis['psi2_2'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='2_2')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train, name='bn2', device=device)
		fm[2] = conv2d(stack_magnitudes(cv4), side_weights['sw2']) #, b=biases['b2_2'])
		
	with tf.name_scope('stage3') as scope:
		cv5 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv5 = complex_input_rotated_conv(cv5, weights['w3_1'], psis['psi3_1'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='3_1')
		cv5 = complex_nonlinearity(cv5, biases['cb3_1'], tf.nn.relu)
	
		cv6 = complex_input_rotated_conv(cv5, weights['w3_2'], psis['psi3_2'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='3_2')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train, name='bn3', device=device)
		fm[3] = conv2d(stack_magnitudes(cv6), side_weights['sw3']) #, b=biases['b3_2'])
		
	with tf.name_scope('stage4') as scope:
		cv7 = mean_pooling(cv6, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv7 = complex_input_rotated_conv(cv7, weights['w4_1'], psis['psi4_1'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='4_1')
		cv7 = complex_nonlinearity(cv7, biases['cb4_1'], tf.nn.relu)
	
		cv8 = complex_input_rotated_conv(cv7, weights['w4_2'], psis['psi4_2'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='4_2')
		cv8 = complex_batch_norm(cv8, tf.nn.relu, phase_train, name='bn4', device=device)
		fm[4] = conv2d(stack_magnitudes(cv8), side_weights['sw4']) #, b=biases['b4_2'])
		
	with tf.name_scope('stage5') as scope:
		cv9 = mean_pooling(cv8, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv9 = complex_input_rotated_conv(cv9, weights['w5_1'], psis['psi5_1'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='5_1')
		cv9 = complex_nonlinearity(cv9, biases['cb5_1'], tf.nn.relu)
	
		cv10 = complex_input_rotated_conv(cv9, weights['w5_2'], psis['psi5_2'],
				 filter_size=3, output_orders=[0,1], padding='SAME', name='5_2')
		cv10 = complex_batch_norm(cv10, tf.nn.relu, phase_train, name='bn5', device=device)
		fm[5] = conv2d(stack_magnitudes(cv10), side_weights['sw5']) #, b=biases['b5_2'])
		
		out = 0
	#with tf.name_scope('classifier') as scope:
	#	out = conv2d(stack_magnitudes(cv10), weights['w_out'], biases['b6'])
	#	out = tf.reduce_mean(out, reduction_indices=[1,2])
	
	fms = {}
	side_preds = []
	with tf.name_scope('fusion') as scope:
		for key in fm.keys():
			if opt['machine'] == 'grumpy':
				fms[key] = tf.image.resize_images(fm[key], tf.pack([size, size2]))
			else:
				fms[key] = tf.image.resize_images(fm[key], size, size2)
			side_preds.append(fms[key])
		side_preds = tf.concat(3, side_preds)

		fms['fuse'] = conv2d(side_preds, side_weights['h1'], b=biases['fuse'], padding='SAME')
		return fms, out

def deep_unet(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = int(opt['n_filters'])
	nf2 = int((opt['filter_gain'])*nf)
	nf3 = int((opt['filter_gain']**2)*nf)
	nf4 = int((opt['filter_gain']**3)*nf)
	bs = opt['batch_size']
	size = opt['dim']
	size2 = opt['dim2']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'd1_1' : get_weights_dict([[3,],[2,]], opt['n_channels'], nf, std_mult=sm, name='d1_1', device=device),
			'd1_2' : get_weights_dict([[3,],[2,]], nf, nf, std_mult=sm, name='d1_2', device=device),
			'd2_1' : get_weights_dict([[3,],[2,],], nf, nf2, std_mult=sm, name='d2_1', device=device),
			'd2_2' : get_weights_dict([[3,],[2,],], nf2, nf2, std_mult=sm, name='d2_2', device=device),
			'd3_1' : get_weights_dict([[3,],[2,],], nf2, nf3, std_mult=sm, name='d3_1', device=device),
			'd3_2' : get_weights_dict([[3,],[2,],], nf3, nf3, std_mult=sm, name='d3_2', device=device),
			'd4_1' : get_weights_dict([[3,],[2,],], nf3, nf4, std_mult=sm, name='d4_1', device=device),
			'd4_2' : get_weights_dict([[3,],[2,],], nf4, nf4, std_mult=sm, name='d4_2', device=device)
		}
		
		psis = {
			'd1_1' : get_phase_dict(1, nf, order, name='d1_1', device=device),
			'd1_2' : get_phase_dict(nf, nf, order, name='d1_2', device=device),
			'd2_1' : get_phase_dict(nf, nf2, order, name='d2_1', device=device),
			'd2_2' : get_phase_dict(nf2, nf2, order, name='d2_2', device=device),
			'd3_1' : get_phase_dict(nf2, nf3, order, name='d3_1', device=device),
			'd3_2' : get_phase_dict(nf3, nf3, order, name='d3_2', device=device),
			'd4_1' : get_phase_dict(nf3, nf4, order, name='d4_1', device=device),
			'd4_2' : get_phase_dict(nf4, nf4, order, name='d4_2', device=device),
			'u1_1' : get_phase_dict(nf4, nf4, order, name='u1_1', device=device),
			'u1_2' : get_phase_dict(nf4, nf4, order, name='u1_2', device=device),
			'u2_1' : get_phase_dict(nf, nf2, order, name='u2_1', device=device),
			'u2_2' : get_phase_dict(nf2, nf2, order, name='u2_2', device=device),
			'u3_1' : get_phase_dict(nf3, nf2, order, name='u3_1', device=device),
			'u3_2' : get_phase_dict(nf2, nf, order, name='u3_2', device=device),
			'u4_1' : get_phase_dict(nf2, nf, order, name='u4_1', device=device),
			'u4_2' : get_phase_dict(nf, nf, order, name='u4_2', device=device)
		}
		
		biases = {
			'b1' : get_bias_dict(nf, order, name='b1', device=device)
		}
		
		x = tf.reshape(x, tf.pack([opt['batch_size'],size,size2,3]))
		fm = {}
		
	# Convolutional Layers
	out, d1 =  down_block(True, x, weights['d1_1'], weights['d1_2'],
						  psis['psi1_1'], psis['psi1_2'], biases['b1'],
						  phase_train, name='down1', device=device)
	out, d2 =  down_block(False, out, weights['d2_1'], weights['d2_2'],
						  psis['psi2_1'], psis['psi2_2'], biases['b2'],
						  phase_train, name='down2', device=device)
	out, d3 =  down_block(False, out, weights['d3_1'], weights['d3_2'],
						  psis['psi3_1'], psis['psi3_2'], biases['b3'],
						  phase_train, name='down3', device=device)
	out, d4 =  down_block(False, out, weights['d4_1'], weights['d4_2'],
						  psis['psi4_1'], psis['psi4_2'], biases['b4'],
						  phase_train, name='down4', device=device)
	
	out =  up_block(out, d4, weights['u4_1'], weights['u4_2'], psis['u4_1'],
					psis['u4_2'], biases['u4'], phase_train, name='up4', device=device)
	out =  up_block(out, d3, weights['u3_1'], weights['u3_2'], psis['u3_1'],
					psis['u3_2'], biases['u3'], phase_train, name='up3', device=device)
	out =  up_block(out, d2, weights['u2_1'], weights['u2_2'], psis['u2_1'],
					psis['u2_2'], biases['u2'], phase_train, name='up2', device=device)
	out =  up_block(out, d1, weights['u1_1'], weights['u1_2'], psis['u1_1'],
					psis['u1_2'], biases['u1'], phase_train, name='up1', device=device)


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

'''
def deep_bsd(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = opt['n_filters']
	nf2 = 2*nf
	nf3 = 4*nf
	nf4 = 8*nf
	bs = opt['batch_size']
	size = int(opt['dim'])
	size2 = int(opt['dim2'])
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1_1' : tf.get_variable('w1_1', dtype=tf.float32, shape=[3,3,3,nf],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w1_2' : tf.get_variable('w1_2', dtype=tf.float32, shape=[3,3,nf,nf],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w2_1' : tf.get_variable('w2_1', dtype=tf.float32, shape=[3,3,nf,nf2],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w2_2' : tf.get_variable('w2_2', dtype=tf.float32, shape=[3,3,nf2,nf2],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w3_1' : tf.get_variable('w3_1', dtype=tf.float32, shape=[3,3,nf2,nf3],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w3_2' : tf.get_variable('w3_2', dtype=tf.float32, shape=[3,3,nf3,nf3],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w4_1' : tf.get_variable('w4_1', dtype=tf.float32, shape=[3,3,nf3,nf4],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w4_2' : tf.get_variable('w4_2', dtype=tf.float32, shape=[3,3,nf4,nf4],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w5_1' : tf.get_variable('w5_1', dtype=tf.float32, shape=[3,3,nf4,nf4],
									 initializer=tf.contrib.layers.variance_scaling_initializer()),
			'w5_2' : tf.get_variable('w5_2', dtype=tf.float32, shape=[3,3,nf4,nf4],
									 initializer=tf.contrib.layers.variance_scaling_initializer())
		}
		
		biases = {
			'b1_1' : tf.get_variable('b1_1', dtype=tf.float32, shape=[nf],
									 initializer=tf.constant_initializer(1e-2)),
			'b1_2' : tf.get_variable('b1_2', dtype=tf.float32, shape=[nf],
									 initializer=tf.constant_initializer(1e-2)),
			'b2_1' : tf.get_variable('b2_1', dtype=tf.float32, shape=[nf2],
									 initializer=tf.constant_initializer(1e-2)),
			'b2_2' : tf.get_variable('b2_2', dtype=tf.float32, shape=[nf2],
									 initializer=tf.constant_initializer(1e-2)),
			'b3_1' : tf.get_variable('b3_1', dtype=tf.float32, shape=[nf3],
									 initializer=tf.constant_initializer(1e-2)),
			'b3_2' : tf.get_variable('b3_2', dtype=tf.float32, shape=[nf3],
									 initializer=tf.constant_initializer(1e-2)),
			'b4_1' : tf.get_variable('b4_1', dtype=tf.float32, shape=[nf4],
									 initializer=tf.constant_initializer(1e-2)),
			'b4_2' : tf.get_variable('b4_2', dtype=tf.float32, shape=[nf4],
									 initializer=tf.constant_initializer(1e-2)),
			'b5_1' : tf.get_variable('b5_1', dtype=tf.float32, shape=[nf4],
									 initializer=tf.constant_initializer(1e-2)),
			'b5_2' : tf.get_variable('b5_2', dtype=tf.float32, shape=[nf4],
									 initializer=tf.constant_initializer(1e-2))
		}
		
		side_weights = {
			'sw1' : tf.get_variable('sw1', dtype=tf.float32, shape=[1,1,nf,1],
					initializer=tf.contrib.layers.variance_scaling_initializer()),
			'sw2' : tf.get_variable('sw2', dtype=tf.float32, shape=[1,1,nf2,1],
					initializer=tf.contrib.layers.variance_scaling_initializer()),
			'sw3' : tf.get_variable('sw3', dtype=tf.float32, shape=[1,1,nf3,1],
					initializer=tf.contrib.layers.variance_scaling_initializer()),
			'sw4' : tf.get_variable('sw4', dtype=tf.float32, shape=[1,1,nf4,1],
					initializer=tf.contrib.layers.variance_scaling_initializer()),
			'sw5' : tf.get_variable('sw5', dtype=tf.float32, shape=[1,1,nf4,1],
					initializer=tf.contrib.layers.variance_scaling_initializer()),
			'h1' : tf.get_variable('h1', dtype=tf.float32, shape=[1,1,5,1],
					initializer=tf.contrib.layers.variance_scaling_initializer()),
		}

		x = tf.reshape(x, tf.pack([opt['batch_size'],size,size2,3]))
		fm = {}
		
	# Convolutional Layers
	with tf.name_scope('stage1') as scope:
		cv1 = conv2d(x, weights['w1_1'], b=biases['b1_1'])
		cv1 = tf.nn.relu(cv1)
	
		cv2 = conv2d(cv1, weights['w1_2'], b=biases['b1_2'])
		cv2 = batch_norm(cv2, phase_train, name='bn1')
		cv2 = tf.nn.relu(cv2)
		fm[1] = conv2d(cv2, side_weights['sw1']) 
	
	with tf.name_scope('stage2') as scope:
		cv2 = tf.nn.max_pool(cv2, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
		cv3 = conv2d(cv2, weights['w2_1'], b=biases['b2_1'])
		cv3 = tf.nn.relu(cv3)
	
		cv4 = conv2d(cv3, weights['w2_2'], b=biases['b2_2'])
		cv4 = batch_norm(cv4, phase_train, name='bn2')
		cv4 = tf.nn.relu(cv4)
		fm[2] = conv2d(cv4, side_weights['sw2'])
		
	with tf.name_scope('stage3') as scope:
		cv4 = tf.nn.max_pool(cv4, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
		cv5 = conv2d(cv4, weights['w3_1'], b=biases['b3_1'])
		cv5 = tf.nn.relu(cv5)
	
		cv6 = conv2d(cv5, weights['w3_2'], b=biases['b3_2'])
		cv6 = batch_norm(cv6, phase_train, name='bn3')
		cv6 = tf.nn.relu(cv6)
		fm[3] = conv2d(cv6, side_weights['sw3']) 
	
	with tf.name_scope('stage4') as scope:
		cv6 = tf.nn.max_pool(cv6, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
		cv7 = conv2d(cv6, weights['w4_1'], b=biases['b4_1'])
		cv7 = tf.nn.relu(cv7)
	
		cv8 = conv2d(cv7, weights['w4_2'], b=biases['b4_2'])
		cv8 = batch_norm(cv8, phase_train, name='bn4')
		cv8 = tf.nn.relu(cv8)
		fm[4] = conv2d(cv8, side_weights['sw4'])
	
	with tf.name_scope('stage5') as scope:
		cv8 = tf.nn.max_pool(cv8, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
		cv9 = conv2d(cv8, weights['w5_1'], b=biases['b5_1'])
		cv9 = tf.nn.relu(cv9)
	
		cv10 = conv2d(cv9, weights['w5_2'], b=biases['b5_2'])
		cv10 = batch_norm(cv10, phase_train, name='bn5')
		cv10 = tf.nn.relu(cv10)
		fm[5] = conv2d(cv10, side_weights['sw5'])
	
	fms = {}
	side_preds = []
	with tf.name_scope('fusion') as scope:
		for key in fm.keys():
			if opt['machine'] == 'grumpy':
				#fms[key] = tf.image.resize_images(fm[key], tf.pack([size, size2]))
				fms[key] = tf.image.resize_bilinear(fm[key], tf.pack([size,size2]),
													align_corners=True)
			else:
				#ms[key] = tf.image.resize_images(fm[key], size, size2)
				fms[key] = tf.image.resize_bilinear(fm[key], size, size2,
													align_corners=True)
			side_preds.append(fms[key])
		side_preds = tf.concat(3, side_preds)

		fms['fuse'] = conv2d(side_preds, side_weights['h1'])
	return fms
'''

##### CUSTOM BLOCKS FOR MODEL #####
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
def pklbatcher(inputs, targets, batch_size, shuffle=False, augment=False,
				img_shape=(321,481,3), anneal=1.):
	"""Input and target are minibatched. Returns a generator"""
	assert len(inputs) == len(targets)
	indices = inputs.keys()
	if shuffle:
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = indices[start_idx:start_idx+batch_size]
		# Data augmentation
		im = []
		targ = []
		for i in xrange(len(excerpt)):
			img = inputs[excerpt[i]]['x']
			tg = targets[excerpt[i]]['y'] > 2
			#tg = tg / np.amax(tg)
			#tg = watershed(tg, anneal)
			if augment:
				# We use shuffle as a proxy for training
				if shuffle:
					img, tg = bsd_preprocess(img, tg)
				#else:
				#	img = img[crop_shape:-crop_shape,crop_shape:-crop_shape,:]
				#	tg = tg[crop_shape:-crop_shape,crop_shape:-crop_shape,:]
			im.append(img)
			targ.append(tg)
		im = np.stack(im, axis=0)
		targ = np.stack(targ, axis=0)
		yield im, targ, excerpt

def imagenet_batcher(data, batch_size, shuffle=False, augment=False):
	"""Input and target are minibatched. Returns a generator"""
	image_dict = dict_to_list(data)
	num_items = len(image_dict.keys())
	indices = np.arange(num_items)
	if shuffle:
		np.random.shuffle(indices)
	for start_idx in range(0, num_items - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = indices[start_idx:start_idx+batch_size]
		# Data augmentation
		im = []
		targ = []
		for i in excerpt:
			img_address = image_dict[indices[i]]['x']
			img = skio.imread(img_address)
			if len(img.shape) == 2:
				img = skco.gray2rgb(img)
			tg = image_dict[indices[i]]['y']
			if augment:
				img = imagenet_preprocess(img)
			else:
				img = imagenet_global_preprocess(img)
			img = central_crop(img, (227,227,3))
			im.append(img)
			targ.append(tg)
		im = np.stack(im, axis=0)
		targ = np.stack(targ, axis=0)
		yield im, targ, excerpt

def dict_to_list(my_dict):
	image_dict = {}
	i = 0
	for key, val in my_dict.iteritems():
		for address in val['addr']:
			image_dict[i] = {}
			image_dict[i]['x'] = address
			image_dict[i]['y'] = val['num']
			i += 1
	return image_dict

def watershed(edge_stack, anneal):
    ''' edge_stack is a HxWxC tensor of all the labels '''
    tmp = distance_transform_edt(1.0 - edge_stack)
    tmp =  np.clip(1.0 - anneal * tmp, 0.0, 1.0)
    return tmp

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
		im = np.concatenate(im, axis=0)
		yield im, targets[excerpt]

def preprocess(im, im_shape, crop_margin):
	'''Data normalizations and augmentations'''
	# Random fliplr
	im = np.reshape(im, im_shape)
	if np.random.rand() > 0.5:
		im = im[:,::-1]
	# Random affine transformation: rotation, scale, stretch, shift and shear
	rdm_scale = log_uniform_rand(1./1.6,1.6)
	new_scale = (rdm_scale*log_uniform_rand(1./1.3,1.3),
				 rdm_scale*log_uniform_rand(1./1.3,1.3))
	new_shear = uniform_rand(-np.pi/9.,np.pi/9.)
	new_angle = uniform_rand(-np.pi, np.pi)
	new_translation = np.asarray((uniform_rand(-crop_margin,crop_margin),
					   uniform_rand(-crop_margin,crop_margin)))
	affine_matrix = sktr.AffineTransform(scale=new_scale, shear=new_shear,
										 translation=new_translation)
	im = sktr.rotate(im, new_angle)
	im = sktr.warp(im, affine_matrix)
	new_shape = np.asarray(im_shape) - 2.*np.asarray((crop_margin,)*2)
	im = central_crop(im, new_shape)
	return np.reshape(im, [1,np.prod(new_shape)])

def bsd_preprocess(im, tg):
	'''Data normalizations and augmentations'''
	fliplr = (np.random.rand() > 0.5)
	flipud = (np.random.rand() > 0.5)
	gamma = np.minimum(np.maximum(1. + np.random.randn(), 0.5), 1.5)
	if fliplr:
		im = np.fliplr(im)
		tg = np.fliplr(tg)
	if flipud:
		im = np.flipud(im)
		tg = np.flipud(tg)
	im = skiex.adjust_gamma(im, gamma)
	return im, tg

def imagenet_global_preprocess(im):
	# Resize
	im = sktr.resize(im, (256,256))
	return ZMUV(im)

def ZMUV(im):
	im = im - np.mean(im, axis=(0,1))
	im = im / np.std(im, axis=(0,1))
	return im

def imagenet_preprocess(im):
	'''Data normalizations and augmentations'''
	# Resize
	im = sktr.resize(im, (256,256))
	# Random numbers
	fliplr = (np.random.rand() > 0.5)
	gamma = np.minimum(np.maximum(1. + np.random.randn(), 0.8), 1.2)
	angle = uniform_rand(0, 360.)
	scale = np.asarray((log_uniform_rand(1/1.1, 1.1), log_uniform_rand(1/1.1, 1.1)))
	translation = np.asarray((uniform_rand(-15,15), uniform_rand(-15,15)))
	# Flips
	if fliplr:
		im = np.fliplr(im)
	# Gamma
	im = skiex.adjust_gamma(im, gamma)
	im = ZMUV(im)
	# Affine transform (no rotation)
	affine_matrix = sktr.AffineTransform(scale=scale, translation=translation)
	im = sktr.warp(im, affine_matrix)
	# Rotate
	im = sktr.rotate(im, angle)
	return im

def central_crop(im, new_shape):
	im_shape = np.asarray(im.shape)
	new_shape = np.asarray(new_shape)
	top_left = (im_shape - new_shape)/2
	bottom_right = top_left + new_shape
	return im[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]

def uniform_rand(min_, max_):
	gap = max_ - min_
	return gap*np.random.rand() + min_

def log_uniform_rand(min_, max_, size=1):
	if size > 1:
		output = []
		for i in xrange(size):
			output.append(10**uniform_rand(np.log10(min_), np.log10(max_)))
	else:
		output = 10**uniform_rand(np.log10(min_), np.log10(max_))
	return output


def save_model(saver, saveDir, sess, saveSubDir=''):
	"""Save a model checkpoint"""
	dir_ = saveDir + "checkpoints/" + saveSubDir
	if not os.path.exists(dir_):
		os.mkdir(dir_)
		print("Created: %s" % (dir_))
	save_path = saver.save(sess, dir_ + "/model.ckpt")
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

##### MAIN SCRIPT #####
def run(opt):
	tf.reset_default_graph()
	# Load dataset
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')
	mnist_trainx, mnist_trainy = mnist_train['x'], mnist_train['y']
	mnist_validx, mnist_validy = mnist_valid['x'], mnist_valid['y']
	mnist_testx, mnist_testy = mnist_test['x'], mnist_test['y']

	# Parameters
	nesterov=True
	model = opt['model']
	lr = opt['lr']
	batch_size = opt['batch_size']
	n_epochs = opt['n_epochs']
	n_filters = opt['n_filters']
	trial_num = opt['trial_num']
	combine_train_val = opt['combine_train_val']
	std_mult = opt['std_mult']
	filter_gain = opt['filter_gain']
	momentum = opt['momentum']
	psi_preconditioner = opt['psi_preconditioner']
	delay = opt['delay']
	model_dir = 'hyperopt_mean_pooling/trial'+str(trial_num)
	
	# Network Parameters
	n_input = 784 				# MNIST data input (img shape: 28*28)
	n_classes = 10 				# MNIST total classes (0-9 digits)
	dataset_size = 10000
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	#keep_prob = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	
	# Construct model
	if model == 'conv_so2':
		pred = conv_so2(x, keep_prob, n_filters, n_classes, batch_size,
						phase_train, std_mult)
	elif model == 'conv_complex_bias':
		pred = conv_complex_bias(x, keep_prob, n_filters, n_classes, batch_size,
								 phase_train, std_mult)
	elif model == 'deep_complex_bias':	
		pred = deep_complex_bias(x, n_filters, n_classes, batch_size,
								 phase_train, std_mult, filter_gain)
	elif model == 'deep_residual':	
		pred = deep_residual(x, n_filters, n_classes, batch_size, phase_train,
							 std_mult)
	else:
		print('Model unrecognized')
		sys.exit(1)
	print('  Using model: %s' % (model,))

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
	opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,
									 momentum=momentum, use_nesterov=nesterov)
	print('  Constructed loss')
	
	grads_and_vars = opt.compute_gradients(cost)
	modified_gvs = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			g = psi_preconditioner*g
		modified_gvs.append((g, v))
	optimizer = opt.apply_gradients(modified_gvs)
	print('  Optimizer built')
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	print('  Evaluation metric constructed')
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	print('  Variables initialized')
	
	if combine_train_val:
		mnist_trainx = np.vstack([mnist_trainx, mnist_validx])
		mnist_trainy = np.hstack([mnist_trainy, mnist_validy])
	
	# Summary writers
	acc_ph = tf.placeholder(tf.float32, [], name='acc_')
	acc_op = tf.scalar_summary("Validation Accuracy", acc_ph)
	cost_ph = tf.placeholder(tf.float32, [], name='cost_')
	cost_op = tf.scalar_summary("Training Cost", cost_ph)
	lr_ph = tf.placeholder(tf.float32, [], name='lr_')
	lr_op = tf.scalar_summary("Learning Rate", lr_ph)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = False
	config.inter_op_parallelism_threads = 1 #prevent inter-session threads?
	sess = tf.Session(config=config)
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
	summary_dir = './logs/hyperopt_mean_pooling/trial'+str(trial_num)
	if not os.path.exists(summary_dir):
		os.mkdir(summary_dir)
		print("Created: %s" % (summary_dir))
	summary = tf.train.SummaryWriter(summary_dir, sess.graph)
	print('  Summaries constructed')
	
	# Launch the graph
	sess.run(init)
	saver = tf.train.Saver()
	epoch = 0
	start = time.time()
	step = 0.
	lr_current = lr
	counter = 0
	best = 0.
	print('  Begin training')
	# Keep training until reach max iterations
	while epoch < n_epochs:
		generator = minibatcher(mnist_trainx, mnist_trainy, batch_size, shuffle=True)
		cost_total = 0.
		acc_total = 0.
		vacc_total = 0.
		for i, batch in enumerate(generator):
			batch_x, batch_y = batch
			#lr_current = lr/np.sqrt(1.+lr_decay*epoch)
			
			# Optimize
			feed_dict = {x: batch_x, y: batch_y, learning_rate : lr_current,
						 phase_train : True}
			__, cost_, acc_ = sess.run([optimizer, cost, accuracy],
				feed_dict=feed_dict)
			if np.isnan(cost_):
				print
				print('Oops: Training went unstable')
				print
				return -1
				
			cost_total += cost_
			acc_total += acc_
			step += 1
		cost_total /=(i+1.)
		acc_total /=(i+1.)
		
		if not combine_train_val:
			val_generator = minibatcher(mnist_validx, mnist_validy, batch_size, shuffle=False)
			for i, batch in enumerate(val_generator):
				batch_x, batch_y = batch
				# Calculate batch loss and accuracy
				feed_dict = {x: batch_x, y: batch_y, phase_train : False}
				vacc_ = sess.run(accuracy, feed_dict=feed_dict)
				vacc_total += vacc_
			vacc_total = vacc_total/(i+1.)
		
		feed_dict={cost_ph : cost_total, acc_ph : vacc_total, lr_ph : lr_current}
		summaries = sess.run([cost_op, acc_op, lr_op], feed_dict=feed_dict)
		for summ in summaries:
			summary.add_summary(summ, step)

		best, counter, lr_current = get_learning_rate(vacc_total, best, counter, lr_current, delay=delay)
		
		print "[" + str(trial_num),str(epoch) + \
			"], Minibatch Loss: " + \
			"{:.6f}".format(cost_total) + ", Train Acc: " + \
			"{:.5f}".format(acc_total) + ", Time: " + \
			"{:.5f}".format(time.time()-start) + ", Counter: " + \
			"{:2d}".format(counter) + ", Val acc: " + \
			"{:.5f}".format(vacc_total)
		epoch += 1
				
		if (epoch) % 50 == 0:
			save_model(saver, './', sess, saveSubDir=model_dir)
	
	print "Testing"
	
	# Test accuracy
	tacc_total = 0.
	test_generator = minibatcher(mnist_testx, mnist_testy, batch_size, shuffle=False)
	for i, batch in enumerate(test_generator):
		batch_x, batch_y = batch
		feed_dict={x: batch_x, y: batch_y, phase_train : False}
		tacc = sess.run(accuracy, feed_dict=feed_dict)
		tacc_total += tacc
	tacc_total = tacc_total/(i+1.)
	print('Test accuracy: %f' % (tacc_total,))
	save_model(saver, './', sess, saveSubDir=model_dir)
	sess.close()
	return tacc_total


if __name__ == '__main__':
	opt = {}
	opt['model'] = 'deep_complex_bias'
	opt['lr'] = 3e-2
	opt['batch_size'] = 53
	opt['n_epochs'] = 120
	opt['n_filters'] = 8
	opt['trial_num'] = 'M'
	opt['combine_train_val'] = False
	opt['std_mult'] = 0.3
	opt['filter_gain'] = 3.7
	opt['momentum'] = 0.93
	opt['psi_preconditioner'] = 3.4
	opt['delay'] = 13
	run(opt)
