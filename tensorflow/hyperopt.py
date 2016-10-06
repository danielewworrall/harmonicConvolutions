'''Hyperparameter optimizer'''

import os
import sys
import time

import numpy as np

from example_scripts import run

def random_independent(n_trials=24):
	y_best = 0.
	best_params = {}
	for i in xrange(n_trials):
		lr = log_uniform_rand(1e-2, 1e-4)
		batch_size = int(log_uniform_rand(25,500))
		n_epochs = int(uniform_rand(100,500))
		n_filters = 10
		batch_norm = None
		print lr, batch_size, n_epochs, n_filters
		y = run(model='conv_so2', lr=lr, batch_size=batch_size,
				n_epochs=n_epochs, n_filters=n_filters,
				bn_config=batch_norm, trial_num=i)
		if y > y_best:
			y_best = y
			best_params['lr'] = lr
			best_params['batch_size'] = batch_size
			best_params['n_epochs'] = n_epochs
			best_params['n_filters'] = n_filters
			best_params['batch_norm'] = batch_norm
		
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
	
	y = []
	for i in xrange(5):
		y.append(run(model='conv_so2', lr=best_params['lr'],
				 batch_size=best_params['batch_size'],
				 n_epochs=best_params['n_epochs'],
				 n_filters=best_params['n_filters'],
				 bn_config=best_params['batch_norm'],
				 trial_num='T-'+str(i), combine_train_val=True))
		print
		print('Current y: %f' % (y[i],))
		print
	
	print('Best y overall')
	print y_best
	print('Best params overall')	
	print best_params
	print y
	y = np.asarray(y)
	mean = np.mean(y)
	print 'Mean: ' + mean
	print 'Std: ' + np.mean((y - mean)**2)

def binary_thinning(n_trials=256):
	y_best = 0.
	best_params = {}
	n_rounds = int(np.log2(n_trials))
	print n_rounds
	
	# Generate parameters
	params = {}
	for trial in xrange(n_trials):
		params[trial] = {}
		params[trial]['lr'] = log_uniform_rand(5e-2, 1e-4)
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
	random_independent(24)
	#binary_thinning(64)
