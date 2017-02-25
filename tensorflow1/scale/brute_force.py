'''Brute force rotation scale'''

import os
import sys
import time

import numpy as np
import tensorflow as tf


def L2_grid(center, shape):
	# Get neighbourhoods
	lin = np.arange(shape)+0.5
	J, I = np.meshgrid(lin, lin)
	I = I - center[1]
	J = J - center[0]
	return np.vstack((np.reshape(I, -1), np.reshape(J, -1)))


def log_polar_filter(N,L,P):
	"""Log-polar transform on a patch by patch basis
	
	N: linear size of filter, assume (N,N) square
	L: the number of radii in the 'log'-direction
	P: the number of orientations in the 'polar'-direction
	"""
	min_radius = 1.
	max_radius = (N/2.)-1
	alpha = np.exp(np.log(max_radius/min_radius)/(L-1.))
	sigma = 1.
	bw = 0.5 #2.*np.pi
	
	# Number of samples on the radial directions
	radii = min_radius*np.power(alpha, np.arange(L))
	# Sample orientations linearly on the circle
	orientations = np.linspace(0, 2.*np.pi, num=P, endpoint=False)
	orientations = np.vstack([-np.sin(orientations), np.cos(orientations)])
	# Need to correct of center of patch
	foveal_center = np.asarray([N, N])/2.
	# Get the coordinates of the samples
	coords = radii[:,np.newaxis,np.newaxis]*orientations[np.newaxis,:,:]
	# Correct for the center of the patch
	# Distance to each pixel. 1) Create a meshgrid 2) Compute squared distances
	mesh = L2_grid(foveal_center, N)[np.newaxis,:,np.newaxis,:]
	dist = coords[:,:,:,np.newaxis] - mesh
	dist2 = np.sum(dist**2, axis=1)
	# Need to filter proportional to radius and num orientations
	bandwidth = bw*radii[:,np.newaxis,np.newaxis] #/P
	weight = np.exp(-dist2/(bandwidth**2)) / (2*np.pi*(bandwidth**2))

	# Gaussian window the filter
	window = np.exp(-np.sum(mesh**2, axis=(1,2)) / ((N/1.)**2) )
	window = window / np.sum(window)
	return np.reshape(weight*window, (-1,N,N)).T


def brute_conv(X, fs, lps, strides=(1,1,1,1), padding='VALID', name='brute'):
	"""Perform a brute force rotation-scale convolution
	
	fs: linear filter shape, scalar
	lps: log polar filter shape (radial, polar)
	"""
	Xsh = X.get_shape().as_list()
	# Convert X to log-polar format
	LPF = log_polar_filter(fs, lps[0], lps[1])
	LPF = np.reshape(LPF, (fs,fs,1,-1))
	LPF = tf.to_float(tf.constant(np.dstack((LPF,)*Xsh[3])))
	# The log-polar transformation
	Y = tf.nn.depthwise_conv2d(X, LPF, strides, padding='SAME', name=name)
	Ysh = Y.get_shape().as_list()
	
	
	return Y


def main():
	x = tf.placeholder(tf.float32, [1,20,20,3])
	y = brute_conv(x, 9, (5,15), name='brute1')
	print y



if __name__ == '__main__':
	main()






































