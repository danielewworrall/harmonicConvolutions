'''Scale conv'''

import os
import sys
import time
sys.path.append('../')

import cv2
import numpy as np
import tensorflow as tf
import skimage.io as skio

import harmonic_network_lite as hn_lite
from harmonic_network_ops import *
from matplotlib import pyplot as plt


def zoom(N, image, factor):
	"""Zoom in on the center of the patch"""
	new_size = (int(factor*image.shape[0]), int(factor*image.shape[1]))
	image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
	new_coords = (int(image.shape[0]*0.5 - (N/2)), int(image.shape[1]*0.5 - (N/2)))
	
	return image[new_coords[0]:new_coords[0]+N, new_coords[1]:new_coords[1]+N]


def s_conv(X, shape, in_range, out_range, name='Sc'):
	# Compute number of orientations
	c0 = 1.
	alpha = 1.1
	n_samples = np.floor((np.log(shape[0]/2) - np.log(c0)) / np.log(alpha))
	radii = c0*np.power(alpha, np.arange(n_samples))
	n_orientations = np.ceil(np.pi*radii[-1])
	# Instantiate the convolutional parameters
	R = get_scale_weights_dict(shape, out_range, 0.4, n_orientations,
		name=name+'_S',device='/gpu:0')
	P = get_phase_dict(shape[2], shape[3], out_range, name=name+'_b',
							 device='/gpu:0')
	W = get_scale_filters(R, shape[0], P=P)
	# The convolution
	return h_range_conv(X, W, in_range=in_range, out_range=out_range,
							  name=name+'_N')


def non_max_suppression(image, size=5):
	"""Stupid non-max suppression. Build boxes and preserve points with highest
	value."""
	from skimage.feature import peak_local_max
	return peak_local_max(image, min_distance=size)


def main():
	"""Run shallow scale conv"""
	fs = 9
	nc = 10
	
	X = skio.imread('../images/balloons.jpg')[50:250,150:350,:]
	x = tf.placeholder(tf.float32, [1,200,200,1,1,3], name='x')
	
	# The convolutions
	s1 = s_conv(x, [fs,fs,3,nc], (0,0), (0,2), name='sc1')
	s1 = hn_lite.nonlinearity(s1, fnc=tf.nn.relu, name='nl1', device='/gpu:0')
	
	s2 = s_conv(s1, [fs,fs,nc,nc], (0,2), (0,2), name='sc2')
	s2 = hn_lite.nonlinearity(s2, fnc=tf.nn.relu, name='nl2', device='/gpu:0')
	
	y = s_conv(s2, [fs,fs,nc,1], (0,2), (0,2), name='sc3')
	mag = stack_magnitudes(y)
	Mags = []
	
	scales = 1 + np.arange(100)*0.04
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		for i in xrange(len(scales)):
			X_zoom = zoom(200, X, scales[i])
			X_ = X_zoom[np.newaxis,:,:,np.newaxis,np.newaxis,:]
			Mag = sess.run(mag, feed_dict={x: X_})
			Mags.append(Mag)
	
	rescaled = []
	for i, m in enumerate(Mags):
		im = np.squeeze(m)[:,:,0]
		factor = 1./scales[i]
		new_size = (int(factor*im.shape[0]), int(factor*im.shape[1]))
		im = cv2.resize(im, new_size, interpolation=cv2.INTER_CUBIC)
		rescaled.append(im)
		
	#plt.imshow(rescaled[0], interpolation='nearest')
	coords = non_max_suppression(rescaled[0], 3)
	plt.scatter(coords[:,1],coords[:,0], color='g')
	#plt.show()	
	
	#plt.ion()
	#plt.show()
	# Repeatability
	sh = rescaled[0].shape
	MSE = []
	for i in xrange(len(rescaled)-1):
		print scales[i]
		shape = rescaled[i+1].shape
		
		#cnew.append(coords[:,1]+(-shape[0]+sh[0])/2,coords[:,0]+(-shape[1]+sh[1])/2)
		
		
		left = ((sh[0]-shape[0])/2, (sh[1]-shape[1])/2)
		right = (left[0]+shape[0], left[1]+shape[1])
		crop = rescaled[0][left[0]:right[0], left[1]:right[1]]
		
		#coords = non_max_suppression(rescaled[i+1], 3)
		'''
		plt.cla()
		plt.imshow(rescaled[0], interpolation='nearest', cmap='gray')
		plt.scatter(coords[:,1]+(-shape[0]+sh[0])/2,coords[:,0]+(-shape[1]+sh[1])/2, color='r')
		plt.xlim(0,200)
		plt.ylim(0,200)
		plt.draw()
		raw_input(i)
		'''
		
		error = rescaled[i+1] - crop
		mse = np.sqrt(np.mean(error**2)/np.mean(np.abs(rescaled[i+1])*np.abs(crop)))
		MSE.append(mse)
		
	plt.plot(scales[1:], MSE)
	plt.show()
	
	
	
	
if __name__ == '__main__':
	main()





































