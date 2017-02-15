'''Scale conv'''

import os
import sys
import time
sys.path.append('../')

import cv2
import numpy as np
import tensorflow as tf
import skimage.io as skio

from harmonic_network_ops import *
from matplotlib import pyplot as plt


def zoom(N, image, factor):
	"""Zoom in on the center of the patch"""
	new_size = (int(factor*image.shape[0]), int(factor*image.shape[1]))
	image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
	new_coords = (int(image.shape[0]*0.5 - (N/2)), int(image.shape[1]*0.5 - (N/2)))
	
	return image[new_coords[0]:new_coords[0]+N, new_coords[1]:new_coords[1]+N]


def main():
	"""Run shallow scale conv"""
	# Compute number of orientations
	fs = 9
	c0 = 1.
	alpha = 1.1
	n_samples = np.floor((np.log(fs/2) - np.log(c0)) / np.log(alpha))
	radii = c0*np.power(alpha, np.arange(n_samples))
	n_orientations = np.ceil(np.pi*radii[-1])
	
	X = skio.imread('../images/balloons.jpg')[50:250,150:350,:]
	#plt.imshow(X)
	#plt.show()
	x = tf.placeholder(tf.float32, [1,200,200,1,1,3], name='x')
	
	# Instantiate the convolutional parameters
	R = get_scale_weights_dict([fs,fs,3,1], (0,2), 0.4, n_orientations,	name='S',
		device='/gpu:0')
	P = get_phase_dict(3, 1, (0,2), name='b',device='/gpu:0')
	W = get_scale_filters(R, fs, P=P)
	
	# The convolution
	y = h_range_conv(x, W, in_range=(0,0), out_range=(0,2), name='N')
	mag = stack_magnitudes(y)
	Mags = []
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		for i in xrange(10):
			X_zoom = zoom(200, X, 1+i*0.1)
			X_ = X_zoom[np.newaxis,:,:,np.newaxis,np.newaxis,:]
			Mag = sess.run(mag, feed_dict={x: X_})
			Mags.append(Mag)
	
	plt.ion()
	plt.show()
	rescaled = []
	for i, m in enumerate(Mags):
		im = np.squeeze(m)[:,:,0]
		factor = 1./(1+i*0.1)
		new_size = (int(factor*im.shape[0]), int(factor*im.shape[1]))
		im = cv2.resize(im, new_size, interpolation=cv2.INTER_CUBIC)
		rescaled.append(im)
		#plt.imshow(im, interpolation='nearest')
		#plt.draw()
		#raw_input(i)
		
	# Now to align and take differences
	
	
if __name__ == '__main__':
	main()





































