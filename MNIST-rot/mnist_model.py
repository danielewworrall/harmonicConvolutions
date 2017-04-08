'''MNIST-rot model'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import tensorflow as tf

import harmonic_network_lite as hn_lite
from harmonic_network_helpers import *

def deep_mnist(opt, x, train_phase, device='/gpu:0'):
	"""The MNIST-rot model similar to the one in Cohen & Welling, 2016"""
	# Sure layers weight & bias
	order = 1
	# Number of Filters
	nf = opt['n_filters']
	nf2 = int(nf*opt['filter_gain'])
	nf3 = int(nf*(opt['filter_gain']**2.))
	bs = opt['batch_size']
	fs = opt['filter_size']
	ncl = opt['n_classes']
	sm = opt['std_mult']

	# Create bias for final layer
	with tf.device(device):
		bias = tf.get_variable('b7', shape=[opt['n_classes']],
							   initializer=tf.constant_initializer(1e-2))
		x = tf.reshape(x, shape=[bs,opt['dim'],opt['dim'],1,1,1])
	
	#with arg_scope([hn_lite.conv2d, hn_lite.non_linearity, hn_lite.batch_norm], device=device):
	# Convolutional Layers with pooling
	with tf.name_scope('block1') as scope:
		cv1 = hn_lite.conv2d(x, nf, fs, padding='SAME', name='1')
		cv1 = hn_lite.non_linearity(cv1, tf.nn.relu, name='1')
		
		cv2 = hn_lite.conv2d(cv1, nf, fs, padding='SAME', name='2')
		cv2 = hn_lite.batch_norm(cv2, train_phase, name='bn1')

	with tf.name_scope('block2') as scope:
		cv2 = hn_lite.mean_pool(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv3 = hn_lite.conv2d(cv2, nf2, fs, padding='SAME', name='3')
		cv3 = hn_lite.non_linearity(cv3, tf.nn.relu, name='3')
		
		cv4 = hn_lite.conv2d(cv3, nf2, fs, padding='SAME', name='4')
		cv4 = hn_lite.batch_norm(cv4, train_phase, name='bn2')

	with tf.name_scope('block3') as scope:
		cv4 = hn_lite.mean_pool(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv5 = hn_lite.conv2d(cv4, nf3, fs, padding='SAME', name='5')
		cv5 = hn_lite.non_linearity(cv5, tf.nn.relu, name='5')
		
		cv6 = hn_lite.conv2d(cv5, nf3, fs, padding='SAME', name='6')
		cv6 = hn_lite.batch_norm(cv6, train_phase, name='bn3')

	# Final Layer
	with tf.name_scope('block4') as scope:
		cv7 = hn_lite.conv2d(cv6, ncl, fs, padding='SAME', phase=False,
					name='7')
		real = hn_lite.sum_magnitudes(cv7)
		cv7 = tf.reduce_mean(real, axis=[1,2,3,4])
		return tf.nn.bias_add(cv7, bias) 