'''Hyperparameter optimizer'''

import os
import sys
import time

import numpy as np

from mnist_tests import run

def main():
	y_best = 0.
	best_params = {}
	n_trials = 24
	for i in xrange(n_trials):
		lr = log_uniform_rand(2e-1, 1e-3)
		batch_size = int(uniform_rand(50, 500))
		n_epochs = int(uniform_rand(100,500))
		n_filters = int(uniform_rand(10,40))
		print lr, batch_size, n_epochs, n_filters
		y = run(model='deep_steer', lr=lr, batch_size=batch_size, 
			n_epochs=n_epochs, n_filters=n_filters)
		if y > y_best:
			y_best = y
			best_params['lr'] = lr
			best_params['batch_size'] = batch_size
			best_params['n_epochs'] = n_epochs
			best_params['n_filters'] = n_filters
		
		print
		print
		print('Best y so far')
		print y_best
		print('Best params so far')	
		print best_params
		print
		print
	
	print('Best y overall')
	print y_best
	print('Best params overall')	
	print best_params
		

def uniform_rand(min_, max_):
	gap = max_ - min_
	return gap*np.random.rand() + min_

def log_uniform_rand(min_, max_):
	return 10**uniform_rand(np.log10(min_), np.log10(max_))

if __name__ == '__main__':
	main()
