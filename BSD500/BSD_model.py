'''BSD model'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import tensorflow as tf

import harmonic_network_lite as hl
from harmonic_network_helpers import *


def to_4d(x):
	"""Convert tensor to 4d"""
	xsh = np.asarray(x.get_shape().as_list())
	return tf.reshape(x, [xsh[0], xsh[1], xsh[2], np.prod(xsh[3:])])

def deep_bsd(opt, x, train_phase, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = int(opt['n_filters'])
	nf2 = int((opt['filter_gain'])*nf)
	nf3 = int((opt['filter_gain']**2)*nf)
	nf4 = int((opt['filter_gain']**3)*nf)
	bs = opt['batch_size']
	fs = opt['filter_size']
	nch = opt['n_channels']
	ncl = opt['n_classes']
	tp = train_phase
	d = device
	
	x = tf.reshape(x, shape=[bs,opt['dim'],opt['dim2'],1,1,3])
	with tf.device(d):
		bias = tf.get_variable('fuse', shape=[1],
							   initializer=tf.constant_initializer(1e-2))
		
		side_weights = {
			'sw1' : tf.get_variable('sw1', shape=[1,1,(order+1)*nf,1],
									initializer=tf.constant_initializer(1e-2)),
			'sw2' : tf.get_variable('sw2', shape=[1,1,(order+1)*nf2,1],
									initializer=tf.constant_initializer(1e-2)),
			'sw3' : tf.get_variable('sw3', shape=[1,1,(order+1)*nf3,1],
									initializer=tf.constant_initializer(1e-2)),
			'sw4' : tf.get_variable('sw4', shape=[1,1,(order+1)*nf4,1],
									initializer=tf.constant_initializer(1e-2)),
			'sw5' : tf.get_variable('sw5', shape=[1,1,(order+1)*nf4,1],
									initializer=tf.constant_initializer(1e-2)),
			'h1' : tf.get_variable('h1', shape=[1,1,5,1],
								   initializer=tf.constant_initializer(1e-2))
		}
		fm = {}
		cv = {}
		
	# Convolutional Layers
	with tf.name_scope('stage1') as scope:
		cv1 = hl.conv2d(x, nf, fs, name='1_1')
		cv1 = hl.non_linearity(cv1, name='1_1')
	
		cv2 = hl.conv2d(cv1, nf, fs, name='1_2')
		cv2 = hl.batch_norm(cv2, tp, name='bn1')
		mags = to_4d(hl.stack_magnitudes(cv2))
		fm[1] = tf.nn.conv2d(mags, side_weights['sw1'], strides=(1,1,1,1), padding='VALID') 
		cv[1] = cv2
	
	with tf.name_scope('stage2') as scope:
		cv3 = hl.mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv3 = hl.conv2d(cv3, nf2, fs, name='2_1')
		cv3 = hl.non_linearity(cv3, name='2_1')
	
		cv4 = hl.conv2d(cv3, nf2, fs, name='2_2')
		cv4 = hl.batch_norm(cv4, train_phase, name='bn2')
		mags = to_4d(hl.stack_magnitudes(cv4))
		fm[2] = tf.nn.conv2d(mags, side_weights['sw2'], strides=(1,1,1,1), padding='VALID') 
		cv[2] = cv4
		
	with tf.name_scope('stage3') as scope:
		cv5 = hl.mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv5 = hl.conv2d(cv5, nf3, fs, name='3_1')
		cv5 = hl.non_linearity(cv5, name='3_1')
	
		cv6 = hl.conv2d(cv5, nf3, fs, name='3_2')
		cv6 = hl.batch_norm(cv6, train_phase, name='bn3')
		mags = to_4d(hl.stack_magnitudes(cv6))
		fm[3] = tf.nn.conv2d(mags, side_weights['sw3'], strides=(1,1,1,1), padding='VALID') 
		cv[3] = cv6
		
	with tf.name_scope('stage4') as scope:
		cv7 = hl.mean_pooling(cv6, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv7 = hl.conv2d(cv7, nf4, fs, name='4_1')
		cv7 = hl.non_linearity(cv7, name='4_1')
	
		cv8 = hl.conv2d(cv7, nf4, fs, name='4_2')
		cv8 = hl.batch_norm(cv8, train_phase, name='bn4')
		mags = to_4d(hl.stack_magnitudes(cv8))
		fm[4] = tf.nn.conv2d(mags, side_weights['sw4'], strides=(1,1,1,1), padding='VALID') 
		cv[4] = cv8
		
	with tf.name_scope('stage5') as scope:
		cv9 = hl.mean_pooling(cv8, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv9 = hl.conv2d(cv9, nf4, fs, name='5_1')
		cv9 = hl.non_linearity(cv9, name='5_1')
	
		cv10 = hl.conv2d(cv9, nf4, fs, name='5_2')
		cv10 = hl.batch_norm(cv10, train_phase, name='bn5')
		mags = to_4d(hl.stack_magnitudes(cv10))
		fm[5] = tf.nn.conv2d(mags, side_weights['sw5'], strides=(1,1,1,1), padding='VALID') 
		cv[5] = cv10
	
	fms = {}
	side_preds = []
	xsh = tf.shape(x)
	with tf.name_scope('fusion') as scope:
		for key in fm.keys():
			#if opt['machine'] == 'grumpy':
			fms[key] = tf.image.resize_images(fm[key], tf.stack([xsh[1], xsh[2]]))
			#else:
			#fms[key] = tf.image.resize_images(fm[key], xsh[1], xsh[2])
			side_preds.append(fms[key])
		side_preds = tf.concat(axis=3, values=side_preds)

		z = tf.nn.conv2d(side_preds, side_weights['h1'], strides=(1,1,1,1), padding='SAME')
		fms['fuse'] = tf.nn.bias_add(z, bias)
		return fms
