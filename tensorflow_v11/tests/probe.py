'''Probe the activations of a network'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import scipy.linalg as scilin
#import tensorflow as tf

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate

import seaborn as sns
sns.set_style("dark")

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


def main():
	x = 485
	y = 1200
	
	data = get_data()
	data = data[x:x+101, y:y+101]
	patch = data #gaussian_filter(data, sigma=.5)

	N = 36
	samples = []
	#plt.ion()
	#plt.show()
	for n in xrange(N):
		angle = (360.*n)/(N-1)
		rot_patch = rotate(patch, angle, reshape=False, order=2)
		rot_patch = rot_patch[45:56,45:56]
		samples.append(interpolate(rot_patch, 1))
		#plt.imshow(rot_patch, interpolation='nearest', cmap='gray', vmin=0.0, vmax=0.6)
		#plt.draw()
		#raw_input()
		
	samples = np.vstack(samples)
	
	plt.ioff()
	y = np.abs(samples)
	x = np.linspace(0, 360., N)
	plt.figure(1)
	c = ['k', 'b', 'g', 'y', 'r']
	for i in xrange(3):
		plt.plot(x, y[:,i], c[i]) #/np.mean(y[:,i]), c[i])
	plt.ylim([0,1.2*np.amax(y)])
	plt.tick_params(axis='both', which='major', labelsize=18)
	plt.xlabel('Rotation (degrees)', fontsize=18)
	plt.ylabel('Magnitude response', fontsize=18)
	plt.tight_layout()
	plt.xlim([0,360])
	#print (np.amax(y, axis=0)-np.amin(y, axis=0)) / np.mean(y, axis=0)
	plt.show()
	
		

def plot_circle(center, radius, k=100):
	x, y = center
	lin = np.linspace(0, 2*np.pi*(k/(k+1.)), k)
	sample_points = (x + radius*np.cos(lin), y + radius*np.sin(lin))
	return np.vstack(sample_points).T


def interpolate(patch, m):
	"""Foveal semi-sampling on rings of length N"""
	# The patch shape determines number of rings and sampling locations. We use a
	# right-hand coordinate system with the first axis pointing down and the
	# second axis going left to right, so (-y,x). We also recenter patches so
	# that the center is (0,0)
	psh = np.asarray(patch.shape)
	min_side = np.minimum(psh[0], psh[1])
	nrings = min_side/2
	radii = np.linspace(m!=0, nrings-0.5, nrings)
	# We define pixel centers to be at positions 0.5
	foveal_center = psh/2.	
	# The angles to sample
	N = np.ceil(np.pi*patch.shape[0])
	lin = (2*np.pi*np.arange(N))/N
	# Sample equi-angularly along each ring
	ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])
	# Create interpolation coefficient coordinates
	coords = L2_grid(foveal_center, psh)
	# Sample positions wrt patch center IJ-coords
	radii = radii[:,np.newaxis,np.newaxis,np.newaxis]
	ring_locations = ring_locations[np.newaxis,:,:,np.newaxis]
	diff = radii*ring_locations - coords[np.newaxis,:,np.newaxis,:]
	dist2 = np.sum(diff**2, axis=1)
	# Convert distances to weightings
	bandwidth = 0.7 #/np.pi
	weights = np.exp(-0.5*dist2/(bandwidth**2))
	# Normalize
	weights = weights/np.sum(weights, axis=2, keepdims=True)
	samples = np.dot(weights, np.reshape(patch, -1))
	
	# Convolve with single frequency
	DFT = scilin.dft(N)[m,:][np.newaxis,:]
	#DFT = np.dot(DFT.conj().T,DFT)/N #########
	LPF = np.dot(DFT, weights)
	LPF_samples = np.dot(LPF, np.reshape(patch, -1))
	
	
	# Plot the foveal center
	
	#plt.figure(1)
	#plt.imshow(patch, interpolation='nearest', cmap='gray')
	#c = ['ro', 'go', 'go', 'yo', 'ro']
	#for i in xrange(len(radii)):
	#	plt.plot(diff[i,0,:,0], diff[i,1,:,0], c[i])
	# Plot the weights for a particular point
	#plt.figure(2)
	#plt.imshow(np.reshape(weights[0,15,:],psh), interpolation='nearest', cmap='gray')
	plt.figure(3)
	LPF_real = np.real(LPF)
	plt.imshow(np.reshape(np.sum(LPF_real, axis=1), (11,11)), interpolation='nearest', cmap='gray')
	plt.figure(4)
	LPF_imag = np.imag(LPF)
	plt.imshow(np.reshape(np.sum(LPF_imag, axis=1), (11,11)), interpolation='nearest', cmap='gray')
	plt.figure(5)
	LPF_abs = LPF_real**2 + LPF_imag**2
	plt.imshow(np.reshape(np.sum(LPF_abs, axis=1), (11,11)), interpolation='nearest', cmap='gray')
	plt.show()
	
	
	return LPF_samples	


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
	main()





































