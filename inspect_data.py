'''Inspect data'''

import os
import sys
import time

import cPickle as pkl
import numpy as np

from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

def watershed(edge_stack):
    ''' edge_stack is a HxWxC tensor of all the labels '''
    tmp = distance_transform_edt(1.0 - edge_stack)
    tmp =  np.clip(1.0 - 1. * tmp, 0.0, 1.0)
    return tmp

def run():
	data_dir = './data/BSR/bsd_pkl'
	with open(data_dir + '/train_images.pkl', 'r') as fp:
		data = pkl.load(fp)
	plt.ion()
	plt.show()
	s = []
	for key, val in data.iteritems():
		#im = val['y'][:,:,0] > 2
		#im = watershed(im)
		im = val['x']
		print np.amax(im)
		print np.amin(im)
		print im.dtype
		plt.imshow(im, interpolation='nearest', cmap='gray')
		plt.draw()
		raw_input(key)
	print np.mean(s)

if __name__ == '__main__':
	run()