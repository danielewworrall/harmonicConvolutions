'''Steerable functions'''

import os
import sys
import time

import cv2
import numpy as np
import skimage.color as skco
import skimage.io as skio

from matplotlib import pyplot as plt
from scipy.linalg import dft
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve as conv
from scipy.signal import tukey
from scipy.misc import imresize


def n_samples(filter_size):
	return np.ceil(np.pi*filter_size)


def L2_grid(center, shape):
	# Get neighbourhoods
	lin = np.arange(shape)+0.5
	J, I = np.meshgrid(lin, lin)
	I = I - center[1]
	J = J - center[0]
	return np.vstack((np.reshape(I, -1), np.reshape(J, -1)))


def log_polar(filter_size, radii, m=0):
	"""Return the filter for the log-polar sinusoid"""
	real = np.cos(m*np.log(radii)) 
	imag = np.sin(m*np.log(radii))
	return (real, imag)


def get_filters(N, m):
	c0 = 1.
	alpha = 1.1
	sigma = 1.
	n_orientations = 25.
	bw = 1.
	
	# Number of samples on the radial directions
	n_samples = np.floor((np.log(N/2) - np.log(c0)) / np.log(alpha))
	radii = c0*np.power(alpha, np.arange(n_samples))
	# Sample orientations linearly on the circle
	orientations = np.linspace(0, 2.*np.pi, num=n_orientations)
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
	bandwidth = bw*radii[:,np.newaxis,np.newaxis]
	weight = np.exp(-dist2/(bandwidth**2)) 
	# The log-radial sinusoids
	lin = np.arange(n_samples)/4.
	sin = np.sin(m*lin) / np.maximum(lin, 1.)
	cos = np.cos(m*lin) / np.maximum(lin, 1.)
	weight_real = np.sum(cos[:,np.newaxis,np.newaxis]*weight, axis=0)
	weight_imag = np.sum(sin[:,np.newaxis,np.newaxis]*weight, axis=0)
	# Gaussian window the filter
	window = np.exp(-np.sum(mesh**2, axis=(1,2)) / (N/3.))
	weight_real = weight_real*window
	weight_imag = weight_imag*window
	
	return weight_real, weight_imag


def zoom(N, image, factor):
	"""Zoom in on the center of the patch"""
	new_size = (int(factor*image.shape[0]), int(factor*image.shape[1]))
	image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
	new_coords = ((image.shape[0] - N)/2, (image.shape[1] - N)/2)
	return image[new_coords[0]:new_coords[0]+N, new_coords[1]:new_coords[1]+N]


def main():
	N = 51
	m = 1
	im = skio.imread('../images/balloons.jpg')[0:200,100:300,:]
	im = skco.rgb2gray(im)
	im = gaussian_filter(im, 1.)
	
	mags = []
	phases = []
	diff = []
	
	mag = []
	phase = []
	weight_real, weight_imag = get_filters(N, m)
	weight_real = np.reshape(np.sum(weight_real, axis=0), (N,N))
	weight_imag = np.reshape(np.sum(weight_imag, axis=0), (N,N))

	for i in xrange(30):		
		scale_factor = 1.+i*0.1
		image = zoom(N, im, scale_factor)
		
		# Inner product
		y_r = np.sum(weight_real*image)
		y_i = np.sum(weight_imag*image)
		# Stats
		phase.append(np.arctan2(y_i, y_r))
		mag.append(np.sqrt(y_r**2 + y_i**2))
		
	plt.figure(1)
	plt.plot(phase)
	#plt.figure(2)
	#plt.plot(mag)
	plt.show()

def one_gauss():
	'''One experiments'''
	from scipy.interpolate import interp1d
	from scipy.signal import resample
	
	N = 100
	signal = np.zeros((N,))
	for i in xrange(N-2):
		signal[i+2] = 0.9*signal[i+1] - 0.3*signal[i] + np.random.randn()
	
	c0 = 1.
	alpha = 1.1
	n_samples = 50
	bw = 1.
	window_size = 45.
	T = window_size / 2
	m = 1.
	beta = 2.
	
	lin = np.arange(n_samples)/4.
	sin = np.sin(m*lin) / np.maximum(lin, 1.)
	cos = np.cos(m*lin) / np.maximum(lin, 1.)
		
	plt.figure(1)
	plt.plot(signal)
	
	phase = []
	mag = []
	for i in xrange(20):
		print 200 + 4*(i+1)
		sig_scale = resample(signal, int(np.floor(200 + 4.*(i+1))))
		length_sig = len(sig_scale)
		log_pos = c0*np.power(alpha, np.arange(n_samples))
		dist2 = (log_pos[:,np.newaxis] - np.arange(length_sig)[np.newaxis,:])**2
		bandwidth = bw*log_pos[:,np.newaxis]
		weight = np.exp(-dist2/(bandwidth**2))
		window = np.exp(-(np.arange(length_sig)/T)**2)
		window[int(window_size):] = 0
		
		weight_real = np.dot(cos, weight)*window
		weight_imag = np.dot(sin, weight)*window
		
		y_r = np.dot(weight_real, sig_scale)
		y_i = np.dot(weight_imag, sig_scale)
		phase.append(np.arctan2(y_i, y_r))
		mag.append(np.sqrt(y_r**2 + y_i**2))
	
		
	plt.figure(2)
	plt.plot(weight_real[:int(2*window_size)])
	plt.plot(weight_imag[:int(2*window_size)])
	plt.plot(window[:int(window_size)])
	plt.figure(3)
	plt.plot(phase)
	plt.figure(4)
	plt.plot(mag)
	plt.show()


if __name__ == '__main__':
	main()
	#one_gauss()












































