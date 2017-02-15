'''Scale conv'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import tensorflow as tf
import skimage.io as skio

from harmonic_network_ops import *


def main():
	"""Run shallow scale conv"""
	# Compute number of orientations
	fs = 9
	c0 = 1.
	alpha = 1.1
	n_samples = np.floor((np.log(fs/2) - np.log(c0)) / np.log(alpha))
	radii = c0*np.power(alpha, np.arange(n_samples))
	n_orientations = np.ceil(np.pi*radii[-1])
	
	X = skio.imread('../images/balloons.jpg')[np.newaxis,...]
	
	x = tf.placeholder(tf.float32, [1,None,None,3], name='x')
	R = get_scale_weights_dict([fs,fs,3,1], (1,3), 0.4, n_orientations,	name='S',
		device='/gpu:0')
	for r in R:
		print r
	print
	W = get_scale_filters(R, fs)
	for w in W:
		print w
	
	
if __name__ == '__main__':
	main()