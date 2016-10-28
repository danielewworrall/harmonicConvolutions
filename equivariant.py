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
def fullyConvolutional(x, drop_prob, n_filters, n_rows, n_cols, n_channels, size_after_conv, n_classes, bs, phase_train, std_mult, filter_gain=2.0, use_batchNorm=True):
	"""The deep_complex_bias architecture. Current test time score is 94.7% for 7 layers 
	deep, 5 filters
	"""
	# Sure layers weight & bias
	order = 3
	nf = n_filters
	nf2 = int(n_filters*filter_gain)
	nf3 = int(n_filters*(filter_gain**2.))
	
	weights = {
		'w1' : get_weights_dict([[6,],[5,],[5,]], n_channels, nf, std_mult=std_mult, name='W1'),
		'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W2'),
		'w3' : get_weights_dict([[6,],[5,],[5,]], nf, nf2, std_mult=std_mult, name='W3'),
		'w4' : get_weights_dict([[6,],[5,],[5,]], nf2, nf2, std_mult=std_mult, name='W4'),
		'w5' : get_weights_dict([[6,],[5,],[5,]], nf2, nf3, std_mult=std_mult, name='W5'),
		'w6' : get_weights_dict([[6,],[5,],[5,]], nf3, nf3, std_mult=std_mult, name='W6'),
		'w7' : get_weights_dict([[6,],[5,],[5,]], nf3, n_classes, std_mult=std_mult, name='W7'),
	}
	
	biases = {
		'b1' : get_bias_dict(nf, 2, name='b1'),
		'b2' : get_bias_dict(nf, 2, name='b2'),
		'b3' : get_bias_dict(nf2, 2, name='b3'),
		'b4' : get_bias_dict(nf2, 2, name='b4'),
		'b5' : get_bias_dict(nf3, 2, name='b5'),
		'b6' : get_bias_dict(nf3, 2, name='b6'),
		'b7' : tf.get_variable('b7', dtype=tf.float32, shape=[n_classes],
			initializer=tf.constant_initializer(1e-2)),
		'psi1' : get_phase_dict(1, nf, 2, name='psi1'),
		'psi2' : get_phase_dict(nf, nf, 2, name='psi2'),
		'psi3' : get_phase_dict(nf, nf2, 2, name='psi3'),
		'psi4' : get_phase_dict(nf2, nf2, 2, name='psi4'),
		'psi5' : get_phase_dict(nf2, nf3, 2, name='psi5'),
		'psi6' : get_phase_dict(nf3, nf3, 2, name='psi6')
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
		if use_batchNorm:
			cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train, outerScope=scope)
	
	with tf.name_scope('block2') as scope:
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)

		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		if use_batchNorm:
			cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train, outerScope=scope)
	
	with tf.name_scope('block3') as scope:
		# LAYER 5
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)

		# LAYER 6
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		if use_batchNorm:
			cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train, outerScope=scope)

	# LAYER 7
	with tf.name_scope('block4') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=5,
								 strides=(1,2,2,1), padding='SAME',
								 name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7'])


def fullyConvolutional_Dieleman(x, drop_prob, n_filters, n_rows, n_cols, n_channels, size_after_conv, n_classes, bs, phase_train, std_mult, filter_gain=-1, use_batchNorm=True):
	"""The conv_so2 architecture, scatters first through an equi_real_conv
	followed by phase-pooling then summation and a nonlinearity. Current
	test time score is 92.97+/-0.06% for 3 layers deep, 15 filters"""
	# Sure layers weight & bias
	order = 3
	nf = n_filters
	
	weights = {
		'w1' : get_weights_dict([[6,],[5,],[5,]], n_channels, 10, std_mult=std_mult, name='W1'),
		'w2' : get_weights_dict([[6,],[5,],[5,]], 10, 10, std_mult=std_mult, name='W2'),
		'w3' : get_weights_dict([[6,],[5,],[5,]], 10, 20, std_mult=std_mult, name='W3'),
		'w4' : get_weights_dict([[6,],[5,],[5,]], 20, 20, std_mult=std_mult, name='W4'),
		'w5' : get_weights_dict([[6,],[5,],[5,]], 20, 40, std_mult=std_mult, name='W5'),
		'w6' : get_weights_dict([[6,],[5,],[5,]], 40, 40, std_mult=std_mult, name='W6'),
		'w7' : get_weights_dict([[6,],[5,],[5,]], 40, 40, std_mult=std_mult, name='W7'),
		'w8' : get_weights_dict([[6,],[5,],[5,]], 40, 60, std_mult=std_mult, name='W8'),
		'w9' : get_weights_dict([[6,],[5,],[5,]], 60, 60, std_mult=std_mult, name='W9'),
		'w10' : get_weights_dict([[6,],[5,],[5,]], 60, 60, std_mult=std_mult, name='W10'),
		'w11' : get_weights_dict([[6,],[5,],[5,]], 60, n_classes, std_mult=std_mult, name='W11'),
	}
	
	biases = {
		'b1' : get_bias_dict(10, 2, name='b1'),
		'b2' : get_bias_dict(10, 2, name='b2'),
		'b3' : get_bias_dict(20, 2, name='b3'),
		'b4' : get_bias_dict(20, 2, name='b4'),
		'b5' : get_bias_dict(40, 2, name='b5'),
		'b6' : get_bias_dict(40, 2, name='b6'),
		'b7' : get_bias_dict(40, 2, name='b7'),
		'b8' : get_bias_dict(60, 2, name='b8'),
		'b9' : get_bias_dict(60, 2, name='b9'),
		'b10' : get_bias_dict(60, 2, name='b10'),
		'b11' : tf.get_variable('b11', dtype=tf.float32, shape=[n_classes],
			initializer=tf.constant_initializer(1e-2)),
		'psi1' : get_phase_dict(n_channels, 10, 2, name='psi1'),
		'psi2' : get_phase_dict(10, 10, 2, name='psi2'),
		'psi3' : get_phase_dict(10, 20, 2, name='psi3'),
		'psi4' : get_phase_dict(20, 20, 2, name='psi4'),
		'psi5' : get_phase_dict(20, 40, 2, name='psi5'),
		'psi6' : get_phase_dict(40, 40, 2, name='psi6'),
		'psi7' : get_phase_dict(40, 40, 2, name='psi7'),
		'psi8' : get_phase_dict(40, 60, 2, name='psi8'),
		'psi9' : get_phase_dict(60, 60, 2, name='psi9'),
		'psi10' : get_phase_dict(60, 60, 2, name='psi10')
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
		if use_batchNorm:
			cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train)
		else:
			cv2 = complex_nonlinearity(cv2, biases['b2'], tf.nn.relu)
	
	with tf.name_scope('block2') as scope:
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)

		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		if use_batchNorm:
			cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train)
		else:
			cv4 = complex_nonlinearity(cv4, biases['b4'], tf.nn.relu)
	
	with tf.name_scope('block3') as scope:
		# LAYER 5
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)

		# LAYER 6
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv6 = complex_nonlinearity(cv6, biases['b6'], tf.nn.relu)

		# LAYER 7
		cv7 = complex_input_rotated_conv(cv6, weights['w7'], biases['psi7'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		if use_batchNorm:
			cv7 = complex_batch_norm(cv7, tf.nn.relu, phase_train)
		else:
			cv7 = complex_nonlinearity(cv7, biases['b7'], tf.nn.relu)

	with tf.name_scope('block4') as scope:
		# LAYER 8
		cv8 = complex_input_rotated_conv(cv7, weights['w8'], biases['psi8'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='5')
		cv8 = complex_nonlinearity(cv8, biases['b8'], tf.nn.relu)

		# LAYER 9
		cv9 = complex_input_rotated_conv(cv8, weights['w9'], biases['psi9'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv9 = complex_nonlinearity(cv9, biases['b9'], tf.nn.relu)

		# LAYER 10
		cv10 = complex_input_rotated_conv(cv9, weights['w10'], biases['psi10'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')

		if use_batchNorm:
			cv10 = complex_batch_norm(cv10, tf.nn.relu, phase_train)
		else:
			cv10 = complex_nonlinearity(cv10, biases['b10'], tf.nn.relu)

	# LAYER 11
	with tf.name_scope('block5') as scope:
		cv11 = complex_input_conv(cv10, weights['w11'], filter_size=5,
								 strides=(1,2,2,1), padding='SAME',
								 name='11')
		cv11 = tf.reduce_mean(sum_magnitudes(cv11), reduction_indices=[1,2])
		return tf.nn.bias_add(cv11, biases['b11'])

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
		cv2 = complex_batch_norm(cv2, lambda x:x, phase_train)
		
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

def get_weights_dict(comp_shape, in_shape, out_shape, std_mult=0.4, name='W'):
	"""Return a dict of weights for use with real_input_equi_conv. comp_shape is
	a list of the number of elements per Fourier base. For 3x3 weights use
	[3,2,2,2]. I currently assume order increasing from 0.
	"""
	weights_dict = {}
	for i, cs in enumerate(comp_shape):
		shape = cs + [in_shape,out_shape]
		weights_dict[i] = get_weights(shape, std_mult=std_mult,
									  name=name+'_'+str(i))
	return weights_dict

def get_bias_dict(n_filters, order, name='b'):
	"""Return a dict of biases"""
	bias_dict = {}
	for i in xrange(order+1):
		bias = tf.get_variable(name+'_'+str(i), dtype=tf.float32, shape=[n_filters],
			initializer=tf.constant_initializer(1e-2))
		bias_dict[i] = bias
	return bias_dict

def get_phase_dict(n_in, n_out, order, name='b'):
	"""Return a dict of phase offsets"""
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

##### MAIN SCRIPT #####
def run(model='', lr=1e-2, batch_size=250, n_epochs=500, n_filters=30, use_batchNorm=True,
		trial_num='N', combine_train_val=False, std_mult=0.4, tf_device='/gpu:0', experimentIdx = 0):
	tf.reset_default_graph()
	#1. LOAD DATA---------------------------------------------------------------------
	if experimentIdx == 0: #MNIST
		print("MNIST")
		# Load dataset
		train = np.load('/home/sgarbin/data/mnist_rotation_new/rotated_train.npz')
		valid = np.load('/home/sgarbin/data/mnist_rotation_new/rotated_valid.npz')
		test = np.load('/home/sgarbin/data/mnist_rotation_new/rotated_test.npz')
		trainx, trainy = train['x'], train['y']
		validx, validy = valid['x'], valid['y']
		testx, testy = test['x'], test['y']

		isClassification = True
		n_rows = 28
		n_cols = 28
		n_channels = 1
		n_input = n_rows * n_cols * n_channels
		n_classes = 10 				# MNIST total classes (0-9 digits)
		size_after_conv = 7 * 7
	elif experimentIdx == 1: #CIFAR10
		print("CIFAR10")
		# Load dataset
		trainx = np.load('/home/sgarbin/data/cifar_numpy/trainX.npy')
		trainy = np.load('/home/sgarbin/data/cifar_numpy/trainY.npy')
		
		validx = np.load('/home/sgarbin/data/cifar_numpy/validX.npy')
		validy = np.load('/home/sgarbin/data/cifar_numpy/validY.npy')

		testx = np.load('/home/sgarbin/data/cifar_numpy/testX.npy')
		testy = np.load('/home/sgarbin/data/cifar_numpy/testY.npy')

		isClassification = True
		n_rows = 32
		n_cols = 32
		n_channels = 3
		n_input = n_rows * n_cols * n_channels
		n_classes = 10 
		size_after_conv = 8 * 8

	elif experimentIdx == 2: #Plankton
		print("Plankton")
		# Load dataset
		trainx = np.load('/home/sgarbin/data/plankton_numpy/trainX.npy')
		trainy = np.load('/home/sgarbin/data/plankton_numpy/trainY.npy')
		
		validx = np.load('/home/sgarbin/data/plankton_numpy/validX.npy')
		validy = np.load('/home/sgarbin/data/plankton_numpy/validY.npy')

		#testx = np.load('/home/sgarbin/data/plankton_numpy/testX.npy')
		#testy = np.load('/home/sgarbin/data/cifar_numpy/testY.npy')

		isClassification = True
		n_rows = 95
		n_cols = 95
		n_channels = 1
		n_input = n_rows * n_cols * n_channels
		n_classes = 121
		size_after_conv = -1
	elif experimentIdx == 3: #Galaxies
		print("Galaxies")
		# Load dataset
		trainx = np.load('/home/sgarbin/data/galaxies_numpy/trainX.npy')
		trainy = np.load('/home/sgarbin/data/galaxies_numpy/trainY.npy')
		
		validx = np.load('/home/sgarbin/data/galaxies_numpy/validX.npy')
		validy = np.load('/home/sgarbin/data/galaxies_numpy/validY.npy')

		#testx = np.load('/home/sgarbin/data/galaxies_numpy/testX.npy')
		#testy = np.load('/home/sgarbin/data/cifar_numpy/testY.npy')

		isClassification = False
		n_rows = 64
		n_cols = 64
		n_channels = 3
		n_input = n_rows * n_cols * n_channels
		n_classes = 37
		size_after_conv = -1
	else:
		print('Dataset unrecognized, options are:')
		print('MNIST: 0')
		print('CIFAR: 1')
		print('PLANKTON: 2')
		print('GALAXIES: 3')
		sys.exit(1)

	#PARAMETERS-----------------------------------------------------------------------
	# Parameters
	lr = lr
	batch_size = batch_size
	n_epochs = n_epochs
	save_step = 100		# Not used yet
	model = model
	
	# Network Parameters
	dropout = 0.75 				# Dropout, probability to keep units
	n_filters = n_filters
	dataset_size = trainx.shape[0] + validx.shape[0]
	print("Total size of trainig set (train + validation): ", dataset_size)
	print("Total output size: ", n_classes)
	if isClassification:
		print("Using classification loss.")
	else:
		print("Using regression loss.")
	print
	print("Learning Rate: ", lr)
	print("Batch Size: ", batch_size)
	print("Num Filters: ", n_filters)
	if use_batchNorm:
		print("Using batchNorm (MomentumOptimiser, m=0.9, learning rate will be annealed)")
	else:
		print("Not using batchNorm (AdamOptimiser, learning rate will not be annealed)")
	#CREATE NETWORK-------------------------------------------------------------------
	# tf Graph input
	with tf.device(tf_device):
		x = tf.placeholder(tf.float32, [batch_size, n_input])
		if isClassification:
			y = tf.placeholder(tf.int64, [batch_size])
		else:
			y = tf.placeholder(tf.float32, [batch_size, n_classes])
		learning_rate = tf.placeholder(tf.float32)
		keep_prob = tf.placeholder(tf.float32)
		phase_train = tf.placeholder(tf.bool)
		
		# Construct model
		if model == 'fullyConvolutional':
			pred = fullyConvolutional(x, keep_prob, n_filters, n_rows, n_cols, n_channels, size_after_conv, n_classes, batch_size, phase_train, std_mult, use_batchNorm)
		elif model == 'fullyConvolutional_Dieleman':
			pred = fullyConvolutional_Dieleman(x, keep_prob, n_filters, n_rows, n_cols, n_channels, size_after_conv, n_classes, batch_size, phase_train, std_mult, use_batchNorm)
		else:
			print('Model unrecognized')
			sys.exit(1)
		print('Using model: %s' % (model,))

		# Define loss and optimizer
		if isClassification:
			cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
		else:
			cost = tf.reduce_sum(tf.pow(y - pred, 2)) / (2 * 37)
		
		if use_batchNorm:
			momentum=0.9
			nesterov=True
			psi_preconditioner = 5e0
			opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=nesterov)
		else:
			opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		# Evaluate model
		if isClassification:
			correct_pred = tf.equal(tf.argmax(pred, 1), y)
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			print('  Evaluation metric constructed')
		else:
			accuracy = cost

		print('  Constructed loss')
		grads_and_vars = opt.compute_gradients(cost)
		modified_gvs = []
		for g, v in grads_and_vars:
			if 'psi' in v.name:
				g = psi_preconditioner*g
			modified_gvs.append((g, v))
		optimizer = opt.apply_gradients(modified_gvs)
		print('  Optimizer built')
		
	grad_summaries_op = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			grad_summaries_op.append(tf.histogram_summary(v.name, g))
			
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
	summary = tf.train.SummaryWriter('./logs/current', sess.graph)
	print('  Summaries constructed')
	
	# GRAPH EXECUTION---------------------------------------------------
	sess.run(init)
	saver = tf.train.Saver()
	epoch = 0
	start = time.time()
	step = 0.
	lr_current = lr
	counter = 0
	best = 0.
	validationAccuracy = -1
	print('  Begin training')
	# Keep training until reach max iterations
	while epoch < n_epochs:
		generator = minibatcher(trainx, trainy, batch_size, shuffle=True)
		cost_total = 0.
		acc_total = 0.
		vacc_total = 0.
		for i, batch in enumerate(generator):
			batch_x, batch_y = batch
			'''print('Test saving...')
			for bi in xrange(batch_size):
					fileName = str(batch_y[bi]) + '_' + str(bi) + '.png'
					sp.misc.imsave(fileName, np.reshape(batch_x[bi, :], (32, 32, 3)))'''
			#lr_current = lr/np.sqrt(1.+lr_decay*epoch)
			
			# Optimize
			feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout,
						 learning_rate : lr_current, phase_train : True}
			__, cost_, acc_, gso = sess.run([optimizer, cost, accuracy,
										grad_summaries_op], feed_dict=feed_dict)
			cost_total += cost_
			acc_total += acc_
			for summ in gso:
				summary.add_summary(summ, step)
			step += 1
		cost_total /=(i+1.)
		acc_total /=(i+1.)
		
		if not combine_train_val:
			val_generator = minibatcher(validx, validy, batch_size, shuffle=False)
			for i, batch in enumerate(val_generator):
				batch_x, batch_y = batch
				'''print('Test saving...')
				for bi in xrange(batch_size):
					fileName = str(batch_y[bi]) + '_' + str(bi) + '.png'
					sp.misc.imsave(fileName, np.reshape(batch_x[bi, :], (32, 32, 3)))'''
				# Calculate batch loss and accuracy
				feed_dict = {x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
				vacc_ = sess.run(accuracy, feed_dict=feed_dict)
				vacc_total += vacc_
			vacc_total = vacc_total/(i+1.)
		
		feed_dict={cost_ph : cost_total, acc_ph : vacc_total, lr_ph : lr_current}
		summaries = sess.run([cost_op, acc_op, lr_op], feed_dict=feed_dict)
		for summ in summaries:
			summary.add_summary(summ, step)

		if use_batchNorm: #only change learning rate when NOT using Adam
			best, counter, lr_current = get_learning_rate(vacc_total, best, counter, lr_current, delay=10)
		
		print "[" + str(trial_num),str(epoch) + \
			"], Minibatch Loss: " + \
			"{:.6f}".format(cost_total) + ", Train Acc: " + \
			"{:.5f}".format(acc_total) + ", Time: " + \
			"{:.5f}".format(time.time()-start) + ", Counter: " + \
			"{:2d}".format(counter) + ", Val acc: " + \
			"{:.5f}".format(vacc_total)
		epoch += 1
		validationAccuracy = vacc_total
				
		if (epoch) % 50 == 0:
			save_model(saver, './', sess)
	
	print "Testing"
	if experimentIdx == 2 or experimentIdx == 3:
		print("Test labels not available for this dataset, relying on validation accuracy instead.")
		print('Test accuracy: %f' % (validationAccuracy,))
		save_model(saver, './', sess)
		sess.close()
		return validationAccuracy
	else:
		# Test accuracy
		tacc_total = 0.
		test_generator = minibatcher(testx, testy, batch_size, shuffle=False)
		for i, batch in enumerate(test_generator):
			batch_x, batch_y = batch
			feed_dict={x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
			tacc = sess.run(accuracy, feed_dict=feed_dict)
			tacc_total += tacc
		tacc_total = tacc_total/(i+1.)
		print('Test accuracy: %f' % (tacc_total,))
		save_model(saver, './', sess)
		sess.close()
		return tacc_total

#ENTRY POINT------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	run(model='deep_complex_bias_NoFC', lr=1e-3, batch_size=100, n_epochs=10, std_mult=0.4,
		n_filters=10, combine_train_val=False)
