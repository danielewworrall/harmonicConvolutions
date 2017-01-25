'''Probe the activations of a network'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
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


def get_kernel(bandwidth, radius, center_offset):
	"""Return the Gaussian kernel"""


def xy2ij(sample_points, imsh):
	J = sample_points[:,0]
	I = imsh[0] - sample_points[:,1]
	return np.vstack((I,J)).T



if __name__ == '__main__':
	multiring((1200,750), np.arange(50)+1)





































