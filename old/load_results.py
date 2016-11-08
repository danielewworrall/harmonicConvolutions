'''Print out all results from a hyperopt'''

import os
import sys
import time

import cPickle as pkl

pkl_dir = './logs/deep_mnist/trial1/'
for root, dir_, file_ in os.walk(pkl_dir):
	for f in file_:
		if '.pkl' in f:
			print pkl_dir + f
			with open(pkl_dir + f, 'r') as fp:
				data = pkl.load(fp)
				print data
				print("%s: %f" % (f, (100.*(1.-data['y']))))
		
