'''Probe the activations of a network'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import scipy.linalg as scilin
#import tensorflow as tf

from matplotlib import pyplot as plt


def get_data():
	import skimage.color as skco
	import skimage.io as skio
	im = skio.imread('../images/scene.jpg')
	return skco.rgb2gray(im)


def multiring(center, radii):
	data = get_data()
	for radius in radii:
		sample_points = plot_circle(center, radius)
		
		# Pick samples from image
		samples = interpolation(data, sample_points)
		# Filter
		samples_FFT = np.fft.fft(samples)
		samples_FFT[3:] *= 0
		# Plot
		plt.figure(2)
		samples_smoothed = np.fft.ifft(samples_FFT)
		plt.plot(np.absolute(samples_smoothed))
	plt.show()


def patch_main():
	x = 400
	y = 1400
	radius = 100
	
	data = get_data()
	patch = data[x:x+11, y:y+11]
	patch_LPF = interpolate(patch, 5)
	

def main():
	x = 1200
	y = 750
	radius = 100
	
	data = get_data()
	sample_points = plot_circle((x, y), radius)
	
	plt.figure(1)
	plt.imshow(data, cmap='gray')
	plt.plot(sample_points[:,0], data.shape[0]-sample_points[:,1], color='r')

	plt.figure(2)
	samples = interpolation(data, sample_points)
	plt.plot(samples)
	
	plt.figure(3)
	samples_FFT = np.fft.fft(samples)
	plt.plot(np.absolute(samples_FFT))
	
	plt.figure(4)
	samples_FFT[3:] *= 0
	samples_smoothed = np.fft.ifft(samples_FFT)
	plt.plot(np.absolute(samples_smoothed))
	plt.show()


def plot_circle(center, radius, k=100):
	x, y = center
	lin = np.linspace(0, 2*np.pi*(k/(k+1.)), k)
	sample_points = (x + radius*np.cos(lin), y + radius*np.sin(lin))
	return np.vstack(sample_points).T


def interpolation(image, sample_points):
	sample_points = xy2ij(sample_points, image.shape)
	neighbourhoods = get_neighbourhoods(sample_points, 4)
	# Get neighbourhoods
	lin = np.arange(2*radius+1)-radius
	X, Y = np.meshgrid(lin, lin)
	tails = sample_points - np.floor(sample_points)
	for i in xrange(tails.shape[0]):
		X_ = X + tails[i,0]
		Y_ = Y + tails[i,1]
		R2 = X_**2 + Y_**2
		neighbourhood = R2 < radius**2
	#sample_points = np.floor(sample_points).astype(np.int32)
	#return image[sample_points[:,0], sample_points[:,1]]


def xy2ij(sample_points, imsh):
	J = sample_points[:,0]
	I = imsh[0] - sample_points[:,1]
	return np.vstack((I,J)).T


def interpolate(patch, m):
	"""Foveal semi-sampling on rings of length N"""
	# The patch shape determines number of rings and sampling locations. We use a
	# right-hand coordinate system with the first axis pointing down and the
	# second axis going left to right, so (-y,x). We also recenter patches so
	# that the center is (0,0)
	psh = np.asarray(patch.shape)
	min_side = np.minimum(psh[0], psh[1])
	nrings = min_side/2
	radii = np.arange(nrings)+0.5*(1+min_side%2)
	if m == 0:
		radii = np.hstack([0,radii])
	# We define pixel centers to be at positions 0.5
	foveal_center = psh/2.
	# The angles to sample
	N = np.ceil(np.pi*patch.shape[0])
	lin = (2*np.pi*np.arange(N))/N
	# Sample equi-angularly along each ring
	ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])
	# Create interpolation coefficient coordinates
	coords = L2_grid(foveal_center, psh)
	#for r in radii:
	# Sample positions wrt patch center IJ-coords
	radii = radii[:,np.newaxis,np.newaxis,np.newaxis]
	ring_locations = ring_locations[np.newaxis,:,:,np.newaxis]
	diff = radii*ring_locations - coords[np.newaxis,:,np.newaxis,:]
	dist2 = np.sum(diff**2, axis=1)
	# Convert distances to weightings
	bandwidth = 1.
	weights = np.exp(-0.5*dist2/(bandwidth**2))
	# Normalize
	weights = weights/np.sum(weights**2, axis=2)[:,:,np.newaxis]
	samples = np.dot(weights, np.reshape(patch, -1))
	
	# Convolve with single frequency
	#dk = dirichlet_kernel(N, 4)
	DFT = scilin.dft(N)[m,:]
	LPF = np.dot(DFT, weights)
	LPF_samples = np.dot(LPF, np.reshape(patch, -1))
	LPF_real = np.reshape(np.real(LPF), np.hstack((-1,psh)))
	for i in xrange(LPF_real.shape[0]):
		plt.figure(i)
		plt.imshow(LPF_real[i,...], cmap='gray', interpolation='nearest')
	plt.show()
	
	'''
	plt.figure(1)
	plt.plot(samples)
	plt.figure(2)
	plt.imshow(patch, interpolation='nearest', cmap='gray', origin='upper')
	for r in np.squeeze(radii):
		x, y = to_image_coords(r*np.squeeze(ring_locations), foveal_center)
		plt.plot(x,y,'ro')
	x, y = to_image_coords(coords, foveal_center)
	#plt.plot(x,y,'bo')
	plt.figure(3)
	for i in xrange(np.squeeze(radii).shape[0]):
		plt.plot(np.real(LPF_samples), 'b')
		plt.plot(np.imag(LPF_samples), 'r')
	plt.show()
	'''


def dirichlet_kernel(N, max_freq):
	"""Return the length-N Dirichlet kernel to lowpass filter on the circle. We
	can easily construct it as an outer product of the bandlimited DFT matrix and
	it's conjugate transpose."""
	DFT = scilin.dft(N)[:max_freq,:]
	return np.dot(DFT.conj().T, DFT)/N
	
		
def to_image_coords(coords, foveal_center):
	x = coords[1,:]+foveal_center[1]-0.5
	y = coords[0,:]+foveal_center[0]-0.5
	return x, y


def L2_grid(center, shape):
	# Get neighbourhoods
	lini = np.arange(shape[0])+0.5
	linj = np.arange(shape[1])+0.5
	J, I = np.meshgrid(lini, linj)
	I = I - center[1]
	J = J - center[0]
	return np.vstack((np.reshape(I, -1), np.reshape(J, -1)))


if __name__ == '__main__':
	patch_main()





































