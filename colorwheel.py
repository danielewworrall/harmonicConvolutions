'''Optic flow color wheel'''

import os

import numpy as np
import skimage.color as skco

from matplotlib import pyplot as plt

def run():
	'''The color wheel'''
	t = np.linspace(-10, 10, num=200)
	x, y = np.meshgrid(t, t)
	rgb = to_optical_flow(x, -y)
	
	plt.imshow(rgb)
	plt.show()

def to_optical_flow(x, y):
	'''Return RGB image using optical flow colorspace'''
	saturation = np.sqrt(x**2 + y**2)
	saturation = 1.5*saturation / np.amax(saturation)
	hue = (np.arctan2(y, x) + np.pi)/(2*np.pi)
	value = np.ones(x.shape)
	
	hsv = np.stack((hue, saturation, value), axis=-1)
	return skco.hsv2rgb(hsv)


if __name__ == '__main__':
	run()