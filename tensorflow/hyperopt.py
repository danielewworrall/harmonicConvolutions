'''Hyperparameter optimizer'''

import os
import sys
import time

import numpy as np

from mnist_tests import run

def random_independent(n_trials=24):
	y_best = 0.
	best_params = {}
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

def binary_thinning(n_trials=256):
	y_best = 0.
	best_params = {}
	n_rounds = int(np.log2(n_trials))
	print n_rounds
	
	# Generate parameters
	params = {}
	for trial in xrange(n_trials):
		params[trial] = {}
		params[trial]['lr'] = log_uniform_rand(2e-1, 1e-3)
		params[trial]['batch_size'] = int(uniform_rand(50, 500))
		params[trial]['n_filters'] = int(uniform_rand(10,40))
	
	# For each trial in list, run experiment
	results = np.zeros((n_trials,))
	sorted_args = np.argsort(-results)
	for j in xrange(n_rounds):
		i = 0
		for trial in sorted_args[:(n_trials/(2**j))]:
			print params[trial]
			params[trial]['y'] = run(model='deep_steer',
									 lr=params[trial]['lr'],
									 batch_size=params[trial]['batch_size'],
									 n_epochs=10*(2**j),
									 n_filters=params[trial]['n_filters'],
									 trial_num=str(j)+'-'+str(i))
			results[trial] = params[trial]['y']
			if params[trial]['y'] > y_best:
				y_best = params[trial]['y']
				best_trial = trial
			print
			print
			print('Best y so far')
			print params[best_trial]
			print
			print
			i += 1
		
		# Sort and reset running best
		sorted_args = np.argsort(-results)
		y_best = 0.
	
	print('Best y in this batch')
	print params[best_trial]

def uniform_rand(min_, max_):
	gap = max_ - min_
	return gap*np.random.rand() + min_

def log_uniform_rand(min_, max_):
	return 10**uniform_rand(np.log10(min_), np.log10(max_))



if __name__ == '__main__':
	binary_thinning(64)
