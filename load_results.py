'''Print out all results from a hyperopt'''

import os
import sys
import time

import cPickle as pkl

pkl_dir = './logs/hyperopt_mean_pooling/numpy/'
for root, dir_, file_ in os.walk(pkl_dir):
	for f in file_:
		with open(pkl_dir + f, 'r') as fp:
			data = pkl.load(fp)
			print("%s: %f" % (f, (100.*(1.-data['y']))))
		
