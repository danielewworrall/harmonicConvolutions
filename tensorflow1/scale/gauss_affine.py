'''Gaussian affine scale-space'''

import os
import sys
import time

import cv2
import numpy as np
import scipy as sp
import skimage.color as skco
import skimage.io as skio

from matplotlib import pyplot as plt
from numpy.linalg import lstsq
from scipy.signal import fftconvolve

#import seaborn as sns
#sns.set_style('whitegrid')

### Generate Gaussian derivative filter ###
def get_filter(N, cov, lim=4):
	precision = np.linalg.inv(cov)
	Z = get_grid(N, lim)
	mahalonobis = np.dot(precision,Z.T)
	mahalonobis = np.sum(mahalonobis*Z.T, axis=0)
	unnormalized = np.exp(-0.5*mahalonobis)
	normalizer = np.sqrt(np.linalg.det(2.*np.pi*cov))
	return np.reshape(unnormalized, (N,N))/normalizer
	

def get_cov(l1,l2,beta):
	rot = np.asarray([[np.cos(beta),-np.sin(beta)],[np.sin(beta),np.cos(beta)]])
	scale = np.asarray([[l1,0],[0,l2]])
	return np.dot(rot,np.dot(scale, rot.T))


def get_grid(N, lim):
	lin = np.linspace(-lim,lim,num=N)
	X, Y = np.meshgrid(lin,lin)
	Y = -Y
	return np.reshape(np.dstack((X,Y)), (-1,2))


def d1(N, cov, phi, lim=4):
	"""First derivative of filter"""
	precision = np.linalg.inv(cov)
	Z = get_grid(N, lim)

	derivative = -np.dot(precision,Z.T).T
	derivative = np.reshape(derivative, (N,N,2))
	return derivative*get_filter(N, cov, lim)[...,np.newaxis]


def get_transformed_filter(N, phi, s1, s2, lim=4):
	direction = np.asarray([np.cos(phi), np.sin(phi)])
	cov1 = get_cov(s1,s2,phi)
	gauss = np.sum(d1(N, cov1, phi, lim)*direction[np.newaxis,np.newaxis,:], axis=2)
	return np.reshape(gauss, -1)

	
### Find filter basis ###
def get_LSQ_filters(N, n_rotations, n_scales, lim=4, freq=None):
	A, P = generate_patches(N, n_rotations, n_scales, lim=lim, freq=freq)
	
	Psi, residuals, rank, s = lstsq(A, P)
	Psi = np.reshape(Psi, (-1,N,N))
	return Psi, residuals, rank, s


def generate_patches(N, n_rotations, n_scales, lim=4, freq=None):
	"""Generate transformed patches"""
	theta = []
	P = []
	#plt.ion()
	#plt.show()
	for i in xrange(n_rotations):
		for j in xrange(n_scales):
			#for k in xrange(n_scales):
			# The transformation parameters
			phi = (2.*np.pi*i)/n_rotations
			s1 = np.power(0.95,j)
			s2 = 1.
			#s2 = np.power(1.05,k)
			# The patch generator
			patch = get_transformed_filter(N, phi, s1, s2, lim=lim)
			# Append data
			theta.append((phi, s1))
			P.append(patch)
	theta = np.vstack(theta)
	# Convert scalings to log
	theta[:,1] = np.log(theta[:,1])
	theta[:,1] -= np.amin(theta[:,1])
	theta[:,1] /= np.amax(theta[:,1])
	theta[:,1] *= np.pi
	'''
	theta[:,2] = np.log(theta[:,2])
	theta[:,2] -= np.amin(theta[:,2])
	theta[:,2] /= np.amax(theta[:,2])
	theta[:,2] *= np.pi
	'''
	A = get_interpolation_function(theta, freq=freq).T
	P = np.vstack(P)
	return A, P


def get_interpolation_function(params, freq=None, N=2):
	M = []
	# The interpolation coefficients are computed as a trigonmetric polynomial
	# of degree N-1 in the transformation variables. If there are K
	# transformation variables, then the product reads
	#
	# P({v}) = sum_{n_1}...sum_{n_K} prod_k [cos(n_k v_k) + sin(n_k v_k)].
	if freq is None:
		freq = []
		for __ in xrange(params.shape[1]):
			freq.append(2*np.arange(N)+1)
	for i in xrange(params.shape[1]):
		A = []
		for m in xrange(N):
			A.append(np.cos(freq[i][m]*params[:,i]))
			A.append(np.sin(freq[i][m]*params[:,i]))
		M.append(np.reshape(np.stack(A), (2*N,-1)))
	W = M[0]
	for i in xrange(1,params.shape[1]):
		W = W[np.newaxis,:,:]*M[i][:,np.newaxis,:]
		W = np.reshape(W, (-1,M[0].shape[-1]))
	return W

'''
def get_sphere_function(params, freq=None, N=2):
	M = []
	# The interpolation coefficients are computed as a trigonmetric polynomial
	# of degree N-1 in the transformation variables. If there are K
	# transformation variables, then the product reads as spherical coordinates.
	if freq is None:
		freq = []
		for __ in xrange(params.shape[1]):
			freq.append(2*np.arange(N)+1)
	
	for i in xrange(len(freqs)):
		pass
	M = single_freq_sphere(params, freqs)
	


def single_freq_sphere(params, freqs):
	M = []
	for i in xrange(params.shape[1]+1):
		A = 1.
		for j in xrange(i):
			A *= np.sin(freqs[j]*params[:,j])
		if i != params.shape[1]:
			A *= np.cos(freqs[i]*params[:,i])
		else:
			A *= np.sin(freqs[i-1]*params[:,i-1])
		M.append(A)
	return np.vstack(M).T
'''

def steer_filter(params, Psi, N=2, freq=None):
	alpha = get_interpolation_function(params, N=N, freq=freq)
	return np.sum(alpha[...,np.newaxis]*Psi, axis=0)


### Experiments ###
def main():
	N = 51
	freq = [[1.,3.],[0.5,1.]]
	Psi, residual, rank, s = get_LSQ_filters(N, 36, 36, lim=3, freq=freq)
	for i in xrange(Psi.shape[0]):
		print np.sum(Psi[i,...]**2)
		plt.subplot(4,4,i+1)
		plt.imshow(Psi[i,...], interpolation='nearest', cmap='jet')
		plt.draw()
		
	plt.ion()
	plt.show()
	plt.cla()
	for rot in np.linspace(0., 2*np.pi, num=36, endpoint=False):
		params = np.asarray([1.,rot/2.])[np.newaxis,:]
		filter_ = steer_filter(params, Psi, freq=freq)
		#filter_ = np.reshape(get_transformed_filter(51, 0., np.power(0.7,rot), 1., lim=2), (51,51))
		plt.imshow(filter_, interpolation='nearest', cmap='jet')
		plt.draw()
		raw_input(rot)


def generate_filters():
	"""Generate filters to be saved for later use"""
	for N in [3,5,7,9,11,13,15,17,19,21]:
		freq = [[1.,3.],[0.5,1.]]
		Psi, residual, rank, s = get_LSQ_filters(N, 36,36, lim=3, freq=freq)
		Psi = Psi / np.sqrt(np.sum(Psi**2, axis=(1,2), keepdims=True))
		'''
		print Psi.shape
		for i in xrange(64):
			plt.subplot(8,8,i+1)
			plt.imshow(Psi[i,...], cmap='jet')
		plt.show()
		'''
		np.save('./filters/aniso/rs_'+str(N)+'.npy', Psi)
	

def response():
	image = skio.imread('../images/balloons.jpg')[50:250,150:350]
	image = skco.rgb2gray(image)
	N = 15
	Psi, residuals, rank, s = get_LSQ_filters(N, 18, 16)
	Theta = np.linspace(0, 2*np.pi, num=360, endpoint=False)
	
	#plt.ion()
	#plt.show()
	for j in xrange(8):
		Y = []
		for theta in Theta:
			M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),180.*theta/(np.pi),1)
			image_ = cv2.warpAffine(image,M,image.shape,flags=cv2.INTER_CUBIC)
			r1 = fftconvolve(image_, Psi[2*j,...], mode='same')
			r2 = fftconvolve(image_, Psi[2*j+1,...], mode='same')
			inv = r1**2 + r2**2
			M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),-180.*theta/(np.pi),1)
			inv = cv2.warpAffine(inv,M,image.shape,flags=cv2.INTER_CUBIC)
			Y.append(inv)
		
		ref = Y[0][50:150,50:150]
		errors = []
		for y in Y:
			crop = y[50:150,50:150]
			error = np.mean((ref-crop)**2) / (np.sqrt(np.mean(ref**2))*np.sqrt(np.mean(crop**2)))
			errors.append(error)
			'''
			plt.imshow(y[50:150,50:150])
			plt.draw()
			raw_input()
			'''
		errors = np.asarray(errors)
		plt.plot(errors)
		plt.tick_params(axis='both', which='major', labelsize=16)
		plt.xlabel('Rotation angle', fontsize=16)
		plt.ylabel('Normalized MSE', fontsize=16)
	plt.show()

if __name__ == '__main__':
	#main()
	#response()
	generate_filters()



























