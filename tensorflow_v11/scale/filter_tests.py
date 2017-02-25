'''Filter tests'''

import os
import sys
import time

import cv2
import numpy as np
import scipy as sp
import skimage.color as skco
import skimage.io as skio

from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

import seaborn as sns
colors = ["windows blue", "amber", "greyish", "faded green"]
colors = sns.xkcd_palette(colors)

def view_filters():
	N_range = 2*np.arange(7) + 3
	plt.ion()
	plt.show()
	for N in N_range:
		print N
		Psi = np.load('./filters/aniso/rs_'+str(N)+'.npy')
		for theta in np.linspace(0, 2.*np.pi, num=36):
			Psi_ = steer_filter(1., theta, Psi)
			plt.imshow(Psi_, interpolation='nearest', cmap='jet')
			plt.draw()
			raw_input()


def steer_filter(theta, scale, Psi):
	params = np.asarray([theta, scale])[np.newaxis,:]
	alpha = get_interpolation_function(params)
	return np.sum(alpha[...,np.newaxis]*Psi, axis=0)


def get_interpolation_function(params, N=2):
	M = []
	# The interpolation coefficients are computed as a trigonmetric polynomial
	# of degree N-1 in the transformation variables. If there are K
	# transformation variables, then the product reads
	#
	# P({v}) = sum_{n_1}...sum_{n_K} prod_k [cos(n_k v_k) + sin(n_k v_k)].
	
	for i in xrange(params.shape[1]):
		A = []
		for m in xrange(N):
			A.append(np.cos((2*m+1)*params[:,i]))
			A.append(np.sin((2*m+1)*params[:,i]))
		M.append(np.reshape(np.stack(A), (2*N,-1)))
	W = M[0]
	for i in xrange(1,params.shape[1]):
		W = W[np.newaxis,:,:]*M[i][:,np.newaxis,:]
		W = np.reshape(W, (-1,M[0].shape[-1]))
	return W


def rotate(image, theta):
	"""Rotate image"""
	imsh = image.shape
	M = cv2.getRotationMatrix2D((imsh[0]/2,imsh[1]/2),180.*theta/(np.pi),1)
	return cv2.warpAffine(image,M,imsh,flags=cv2.INTER_CUBIC)


def xy_scale(image, scale=(1.,1.), new_shape=None):
	"""Anisotropic scaling in x-y. If new_shape is given use that over scale"""
	imsh = image.shape
	if new_shape is None:
		new_shape = (int(imsh[0]*scale[0]),int(imsh[1]*scale[1]))
	return cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)


def get_mag_errors(Y, crop_region=np.s_[50:150,50:150]):
	"""Return the relative squared errors in the magnitude maps"""
	ref = Y[0][crop_region]
	errors = []
	for y in Y:
		crop = y[crop_region]
		error = np.mean((ref-crop)**2)
		error /= (np.sqrt(np.mean(ref**2))*np.sqrt(np.mean(crop**2)))
		errors.append(np.sqrt(error))
	return np.asarray(errors)


def magnitude_response():
	image = skio.imread('../images/balloons.jpg')[50:250,150:350]
	image = skco.rgb2gray(image)
	imsh = image.shape
	
	Theta = np.linspace(0, 2*np.pi, num=360, endpoint=False)
	N_range = 2*np.arange(10) + 3
	
	plt.ion()
	plt.show()
	max_errors = []
	for N in N_range:
		print N
		Psi = np.load('./filters/fractional_orders/rs_'+str(N)+'.npy')
		max_error = []
		for color_num, j in enumerate([0,2,8,10]):
			Y = []
			for theta in Theta:
				# Transform
				#image_ = rotate(image, theta)
				image_ = xy_scale(image, scale=(np.power(1.132,theta), 1.))
				
				# Convolve
				inv = 0.
				for k in [0,1,4,5]:
					inv += fftconvolve(image_, Psi[j+k,...], mode='same')**2
				inv = np.sqrt(inv)
				
				# Inverse transform
				#inv = rotate(inv, -theta)
				inv = xy_scale(inv, new_shape=imsh)
				Y.append(inv)
			
			# Crop out center of image
			errors = get_mag_errors(Y, crop_region=np.s_[50:150,50:150])
			max_error.append(np.amax(errors))
		max_errors.append(np.hstack(max_error))
	
	plt.plot(N_range, max_errors)
	plt.xlabel('Filter size')
	plt.ylabel('Maximum error')
	plt.draw()
	raw_input()


def get_phase_errors(phase, Theta):
	"""Return the deviation from linear phase"""
	phase = np.hstack(phase)
	# Find jumps and rectify
	phase_diff = np.append(np.diff(phase),phase[0]-phase[-1])
	phase_diff[np.abs(phase_diff) > 5.5] = Theta[1] - Theta[0]
	phase = np.cumsum(phase_diff)
	# Measure deviation
	deviations = []
	deviations.append(phase-Theta)
	deviations.append(phase+Theta)
	# Frequency 3
	phase_diff3 = np.append(np.diff(phase),phase[0]-phase[-1])
	phase_diff3[np.abs(phase_diff3) > 5.5] = Theta[1] - Theta[0]
	phase3 = np.cumsum(phase_diff3)
	deviations.append(phase3-3.*Theta)
	
	max_dev = []
	for i in xrange(len(deviations)):
		max_dev.append(np.amax(np.abs(np.diff(deviations[i]))))
	min_ = np.argmin(max_dev)
	
	plt.ioff()
	plt.plot(phase)
	plt.plot(deviations[0])
	plt.plot(deviations[1])
	plt.plot(deviations[2])
	plt.show()
	
	return np.abs(deviations[min_])


def phase_response():
	image = skio.imread('../images/balloons.jpg')[50:250,150:350]
	image = skco.rgb2gray(image)
	imsh = image.shape
	
	Theta = np.linspace(0, 2*np.pi, num=360, endpoint=False)
	N_range = 2*np.arange(7) + 15
	
	plt.ion()
	plt.show()
	max_deviations = []
	for N in N_range:
		print N
		Psi = np.load('./filters/rs_'+str(N)+'.npy')
		max_deviation = []
		for color_num, j in enumerate([0,1,2,3,8,9,10,11]):
			phases = []
			for theta in Theta:
				# Transform
				#image_ = rotate(image, theta)
				image_ = xy_scale(image, scale=(1., np.power(1.132,theta)))
				
				# Convolve
				c = fftconvolve(image_, Psi[j,...], mode='same')
				s = fftconvolve(image_, Psi[j+4,...], mode='same')
				phase = np.arctan2(s,c)
				
				# Inverse transform
				#phase = rotate(phase, -theta)
				phase = xy_scale(phase, new_shape=imsh)
				phases.append(phase[100,100])
				
				plt.imshow(phase, cmap='jet')
				plt.draw()
				raw_input()
			
			# Calculate absolute deviations in phase
			#abs_deviations = np.abs(get_phase_errors(phases, Theta))
			#max_deviation.append(np.amax(abs_deviations))
		#max_deviations.append(np.hstack(max_deviation))
	
	plt.plot(N_range, max_deviations)
	plt.xlabel('Filter size')
	plt.ylabel('Maximum deviation (rad)')
	plt.draw()
	raw_input()





if __name__ == '__main__':
	magnitude_response()
	#phase_response()
	#view_filters()






























