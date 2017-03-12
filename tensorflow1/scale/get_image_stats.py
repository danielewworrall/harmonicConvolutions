'''Get stats of images'''

import os
import sys
import time

import numpy as np
import skimage.io as skio

from matplotlib import pyplot as plt


def get_stats():
	root = '/home/dworrall'
	folder = root + '/Data/faces/images'
	
	n_images = 0
	running_mean = 0.
	running_var = 0.
	
	# Get mean
	print('Getting means')
	for root, dirs, files in os.walk(folder):
		for f in files:
			if '.png' in f:
				fname = '{:s}/{:s}'.format(root, f)
				image = skio.imread(fname)
				running_mean = update_stat(image, running_mean, n_images)
				n_images += 1
				if n_images % 10 == 0:
					sys.stdout.write('{:d}\r'.format(n_images))
					sys.stdout.flush()
	np.save('/home/dworrall/Data/faces/mean.npy', running_mean)
	
	# Get variance
	print('Getting stddevs')
	for root, dirs, files in os.walk(folder):
		for f in files:
			if '.png' in f:
				fname = '{:s}/{:s}'.format(root, f)
				image = skio.imread(fname)
				running_var = update_stat((image-running_mean)**2, running_var, n_images)
				n_images += 1
				if n_images % 10 == 0:
					sys.stdout.write('{:d}\r'.format(n_images))
					sys.stdout.flush()
	np.save('/home/dworrall/Data/faces/stddev.npy', np.sqrt(running_var))


def update_stat(current_stat, running_stat, num_images):
	"""Update running mean"""
	weight = num_images/(1.+num_images)
	return (1.-weight)*current_stat + weight*running_stat
	

if __name__ == '__main__':
	get_stats()