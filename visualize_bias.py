'''Visualize the compelx phase'''

import os
import sys
import time

import cv2
import numpy as np
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import tensorflow as tf

import input_data

from steer_conv import *

from matplotlib import pyplot as plt

def get_weights_dict(comp_shape, in_shape, out_shape, std_mult=0.4, name='W'):
	"""Return a dict of weights for use with real_input_equi_conv. comp_shape is
	a list of the number of elements per Fourier base. For 3x3 weights use
	[3,2,2,2]. I currently assume order increasing from 0.
	"""
	weights_dict = {}
	for i, cs in enumerate(comp_shape):
		shape = cs + [in_shape,out_shape]
		weights_dict[i] = get_weights(shape, std_mult=std_mult, name=name+'_'+str(i))
	return weights_dict

def get_bias_dict(n_filters, const, order, name='b'):
	"""Return a dict of biases"""
	bias_dict = {}
	for i in xrange(order+1):
		bias_dict[i] = const
	return bias_dict

def get_complex_bias_dict(n_filters, order, name='b'):
	"""Return a dict of biases"""
	bias_dict = {}
	for i in xrange(order+1):
		bias_x = tf.Variable(tf.constant(1e-2, shape=[n_filters]), name=name+'x_'+str(i))
		bias_y = tf.Variable(tf.constant(1e-2, shape=[n_filters]), name=name+'y_'+str(i))
		bias_dict[i] = (bias_x, bias_y)
	return bias_dict


##### MAIN SCRIPT #####
def run():
	tf.reset_default_graph()
	
	const = tf.placeholder(tf.float32, [])
	
	R = get_weights_dict([[6,],[5,],[5,]], 1, 1, std_mult=0.4)
	psi = get_bias_dict(1, const, 2)
	Q = get_complex_rotated_filters(R, psi, filter_size=5)
	
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Launch the graph
	plt.ion()
	plt.show()
	with tf.Session() as sess:
		sess.run(init)
		for c in np.linspace(0,2*np.pi, num=36):
			Q_ = sess.run(Q, feed_dict={const : c})
			plt.figure(1)
			plt.imshow(np.squeeze(Q_[2][0]), cmap='gray', interpolation='nearest')
			plt.figure(2)
			plt.imshow(np.squeeze(Q_[2][1]), cmap='gray', interpolation='nearest')
			plt.draw()
			raw_input(c)

if __name__ == '__main__':
	run()
