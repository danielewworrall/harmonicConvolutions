'''Read hyperopt output'''

import os
import sys
import time
sys.path.append('../')

import cPickle as pkl
import tensorflow as tf

import harmonic_network_models

folder = 'results'
for root, dirs, files in os.walk(folder):
	for f in files:
		fname = root + '/' + f
		with open(fname, 'r') as fp:
			data = pkl.load(fp)
		for k, v in data.iteritems():
			print k, v
		print
		print