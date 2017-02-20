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
def get_filter(N, cov):
	precision = np.linalg.inv(cov)
	Z = get_grid(N)
	mahalonobis = np.dot(precision,Z.T)
	mahalonobis = np.sum(mahalonobis*Z.T, axis=0)
	unnormalized = np.exp(-0.5*mahalonobis)
	normalizer = np.sqrt(np.linalg.det(2.*np.pi*cov))
	return np.reshape(unnormalized, (N,N))/normalizer
	

def get_cov(l1,l2,beta):
	rot = np.asarray([[np.cos(beta),-np.sin(beta)],[np.sin(beta),np.cos(beta)]])
	scale = np.asarray([[l1,0],[0,l2]])
	return np.dot(rot,np.dot(scale, rot.T))


def get_grid(N):
	lin = np.linspace(-5,5,num=N)
	X, Y = np.meshgrid(lin,lin)
	Y = -Y
	return np.reshape(np.dstack((X,Y)), (-1,2))


def d1(N, cov, phi):
	"""First derivative of filter"""
	precision = np.linalg.inv(cov)
	Z = get_grid(N)

	derivative = -np.dot(precision,Z.T).T
	derivative = np.reshape(derivative, (N,N,2))
	return derivative*get_filter(N, cov)[...,np.newaxis]


def get_transformed_filter(N, phi, s1, s2):
	direction = np.asarray([np.cos(phi), np.sin(phi)])
	cov1 = get_cov(s1,s2,phi)
	gauss = np.sum(d1(N, cov1, phi)*direction[np.newaxis,np.newaxis,:], axis=2)
	return np.reshape(gauss, -1)

	
### Find filter basis ###
def get_LSQ_filters(N, n_rotations, n_scales, ):
	theta = []
	P = []
	for i in xrange(n_rotations):
		for j in xrange(n_scales):
			for k in xrange(n_scales):
				# The transformation parameters
				phi = (2.*np.pi*i)/n_rotations
				s1 = np.power(1.05,j)
				s2 = np.power(1.05,k)
				# The patch generator
				patch = get_transformed_filter(N, phi, s1, s2)
				# Append data
				theta.append((phi, s1, s2))
				P.append(patch)
	theta = np.vstack(theta)
	# Convert scalings to log
	theta[:,1] = np.log(theta[:,1])
	theta[:,1] -= np.amin(theta[:,1])
	theta[:,1] /= np.amax(theta[:,1])
	theta[:,1] *= np.pi
	
	theta[:,2] = np.log(theta[:,2])
	theta[:,2] -= np.amin(theta[:,2])
	theta[:,2] /= np.amax(theta[:,2])
	theta[:,2] *= np.pi
	
	A = get_interpolation_function(theta).T
	P = np.vstack(P)
	
	Psi, residuals, rank, s = lstsq(A, P)
	Psi = np.reshape(Psi, (-1,N,N))
	return Psi, residuals, rank, s


def get_interpolation_function(params):
	N = 3
	M = []
	# The interpolation coefficients are computed as a trigonmetric polynomial
	# of degree N-1 in the transformation variables. If there are K
	# transformation variables, then the product reads
	#
	# P({v}) = sum_{n_1}...sum_{n_K} prod_k [cos(n_k v_k) + sin(n_k v_k)].
	
	for i in xrange(params.shape[1]):
		A = []
		for m in xrange(N):
			A.append(np.cos(m*params[:,i]))
			A.append(np.sin(m*params[:,i]))
		M.append(np.reshape(np.stack(A), (2*N,-1)))
	W = M[0]
	for i in xrange(1,params.shape[1]):
		W = W[np.newaxis,:,:]*M[i][:,np.newaxis,:]
		W = np.reshape(W, (-1,M[0].shape[-1]))
	return W


def steer_filter(theta, scale, Psi):
	params = np.asarray([theta, scale, scale])[np.newaxis,:]
	alpha = get_interpolation_function(params)
	return np.sum(alpha[...,np.newaxis]*Psi, axis=0)


### Experiments ###
def main():
	N = 51
	Psi, __, rank, s = get_LSQ_filters(N, 72, 16)
	P = np.reshape(Psi, (-1,N*N))
	
	plt.ion()
	plt.show()
	for i in xrange(16):
		print np.sum(Psi[i,...]**2)
		plt.imshow(Psi[i,...], interpolation='nearest')
		plt.draw()
		raw_input()
		
	plt.ion()
	plt.show()
	plt.cla()
	for rot in np.linspace(0., 2*np.pi, num=36, endpoint=False):
		filter_ = steer_filter(1., rot/2., Psi)
		plt.imshow(filter_, interpolation='nearest')
		plt.draw()
		raw_input(rot)


def response():
	image = skio.imread('../images/balloons.jpg')[50:250,150:350]
	image = skco.rgb2gray(image)
	N = 9
	Psi, residuals, rank, s = get_LSQ_filters(N, 72, 16)
	Theta = np.linspace(0, 2*np.pi, num=360, endpoint=False)
	
	for j in xrange(20): #Psi.shape[0]/2):
		Y = []
		if np.sum(Psi[2*j,...]**2) > 1e-5:
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
				#plt.imshow(y[50:150,50:150])
				#plt.draw()
				#raw_input()
			errors = np.asarray(errors)
			plt.plot(errors)
			plt.tick_params(axis='both', which='major', labelsize=16)
			plt.xlabel('Rotation angle', fontsize=16)
			plt.ylabel('Normalized MSE', fontsize=16)
			#plt.tight_layout()
			#plt.draw()
			#raw_input()
	plt.show()

if __name__ == '__main__':
	#main()
	response()




























