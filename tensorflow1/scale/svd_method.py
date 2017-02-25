'''SVD method'''

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
def get_LSQ_filters(N, n_rotations, n_scales, lim=4):
	A, P = generate_patches(N, n_rotations, n_scales, lim=4)
	
	Psi, residuals, rank, s = lstsq(A, P)
	Psi = np.reshape(Psi, (-1,N,N))
	return Psi, residuals, rank, s


def generate_patches(N, n_rotations, n_scales, lim=4):
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
			s1 = np.power(1.05,j)
			#s2 = np.power(1.05,k)
			# The patch generator
			s2 = s1
			patch = get_transformed_filter(N, phi, s1, s2, lim=lim)
			# Append data
			theta.append((phi, s1)) #, s2))
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
	A = get_interpolation_function(theta).T
	P = np.vstack(P)
	return A, P


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


def steer_filter(params, Psi, N=2):
	alpha = get_interpolation_function(params, N=N)
	return np.sum(alpha[...,np.newaxis]*Psi, axis=0)


def svd_method():
	"""Perona's SVD method"""
	N = 51
	n_orientations = 32
	n_scales = 16
	A, P = generate_patches(N, n_orientations, n_scales, lim=4)
	U, s, V = np.linalg.svd(P)
	
	plt.ion()
	plt.show()
	plt.plot(np.cumsum(s[:50]**2) / np.sum(s**2))
	raw_input()
	plt.clf()
	for i in xrange(36):
		plt.imshow(np.reshape(V[i,:], (N,N)), interpolation='nearest')
		plt.draw()
		raw_input(i)


def approximation_error():
	"""Find the error with the SVD approximation"""
	N = 15
	n_orientations = 72
	n_scales = 16
	A, P = generate_patches(N, n_orientations, n_scales, lim=4)
	U, s, V = np.linalg.svd(P)
	patch = P[405,...]
	
	plt.imshow(np.dot(V, V.T), cmap='jet')
	plt.show()
	coeffs = np.dot(V, patch)
	errors = []
	for i in xrange(64):
		recon = np.dot(coeffs[:i],V[:i,:])
		errors.append(np.sqrt(np.sum((recon - patch)**2)))
	plt.plot(errors)
	plt.show()


def pairs():
	"""Find the error with the SVD approximation"""
	N = 51
	K = 10
	n_orientations = 16
	n_scales = 16
	A, P = generate_patches(N, n_orientations, n_scales, lim=4)
	U, s, V = np.linalg.svd(P)
	
	V = V[:K,:]
	V2 = V**2
	pairs = V2[np.newaxis,:,:] + V2[:,np.newaxis,:]
	pairs = np.reshape(pairs, (-1,N,N))
	
	for i in xrange(K**2):
		plt.subplot(K, K, i+1)
		plt.imshow(pairs[i,...], cmap='jet')
		plt.axis('off')
	plt.show()
	

def natural_patches():
	"""SVD on real patches"""
	folder = './trevi/'
	
	N = 2000
	patches = []
	for i in xrange(395):
		fname = folder + 'patches{:04d}.bmp'.format(i)
		image = skio.imread(fname) / 255.
		for j in xrange(16):
			for k in xrange(16):
				patches.append(np.reshape(image[64*j:64*(j+1),64*k:64*(k+1)], -1))
	patches = np.vstack(patches)
	choice = np.random.choice(patches.shape[0], N, replace=False)
	patches = patches[choice,:]
	print('Patches loaded')
	
	U, s, V = np.linalg.svd(patches)
	plt.plot(np.cumsum(s[:50]**2) / np.sum(s**2))
	plt.show()
	
	K = 20
	for i in xrange(K**2):
		plt.subplot(K, K, i+1)
		plt.imshow(np.reshape(V[i,...], (64,64)))
		plt.axis('off')
	plt.show()


if __name__ == '__main__':
	#svd_method()
	#approximation_error()
	#pairs()
	natural_patches()
































