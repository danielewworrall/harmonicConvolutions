'''View rotation-scale filters'''

import os
import sys
import time

import numpy as np
import scipy.signal as scisig
import skimage.color as skco
import skimage.io as skio
import skimage.transform as sktr

from matplotlib import pyplot as plt


def convolve(m, n, size):
	im = skio.imread('./scale.png')
	im = skco.rgb2gray(im)
	
	comp = return_filter(m, n, size)
	filt = []
	results = []
	
	filt.append(comp[0]*comp[2]-comp[1]*comp[3])
	filt.append(comp[0]*comp[3]+comp[1]*comp[2])
	#filt.append(comp[2])
	#filt.append(comp[3])
	
	for i in xrange(2):
		fig = plt.figure(i)
		results.append(scisig.fftconvolve(im, filt[i]))
		plt.imshow(results[i], cmap='gray')
	plt.figure(i+1)
	mag = np.sqrt(results[0]**2 + results[1]**2)
	plt.imshow(mag, cmap='gray')
	plt.show()


def return_filter(m, n, size):
	radius, theta = get_grid(size)
	real, imag = get_rotation_component(m, theta)
	radial_real, radial_imag = get_radial_component(n, radius)
	return [real, imag, radial_real, radial_imag]


def view_filter(m, n, size):
	"""View filters"""
	comp = return_filter(m, n, size)
	filt = []
	filt.append(comp[0]*comp[2]-comp[1]*comp[3])
	filt.append(comp[0]*comp[3]+comp[1]*comp[2])
	#filt.append(comp[2])
	#filt.append(comp[3])
	
	for i in xrange(2):
		fig = plt.figure(i)
		plt.imshow(filt[i], cmap='gray', interpolation='nearest')
	plt.show()


def get_rotation_component(order, theta):
	"""Plot the rotation component of the filters"""
	return np.cos(order*theta), np.sin(order*theta)


def get_radial_component(order, radius):
	"""Plot the rotation component of the filters"""
	radius = np.maximum(radius, 1e-9)
	lr = np.log(radius)
	return np.cos(order*lr), np.sin(order*lr)


def get_grid(size):
	"""Get a polar grid"""
	linspace = np.reshape(np.arange(size) - size/2, (size,1))
	ones = np.ones((size,1))
	X = np.dot(ones,linspace.T)
	Y = -X.T
	theta = np.arctan2(Y,X)
	radius = np.sqrt(X**2 + Y**2)
	return radius, theta


if __name__ == '__main__':
	view_filter(1,-2,125)
	#convolve(1,2,25)