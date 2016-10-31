'''Hyperparameter optimizer'''

import os
import sys
import shutil
import time

import cPickle as pkl
import numpy as np

from equivariant import run

def random_independent(n_trials=3):
	y_best = 0.
	best_params = {}
	opt = {}
	for i in xrange(n_trials):
		opt['y'] = -1
		while opt['y'] < 0:
			# Remove files corresponding to trial
			trial_num = i + 10
			log_path = './logs/hyperopt_mean_pooling/trial'+str(trial_num)
			if os.path.exists(log_path):
				shutil.rmtree(log_path)
			checkpoint_path = './checkpoints/hyperopt_mean_pooling/trial'+str(trial_num)
			if os.path.exists(checkpoint_path):
				shutil.rmtree(checkpoint_path)
			
			opt['model'] = 'deep_complex_bias'
			opt['lr'] = log_uniform_rand(5e-3, 1e-1)
			opt['batch_size'] = int(log_uniform_rand(40,120))
			opt['n_epochs'] = 100
			opt['n_filters'] = int(uniform_rand(2,12))
			opt['trial_num'] = trial_num
			opt['combine_train_val'] = False
			opt['std_mult'] = uniform_rand(0.05, 0.5)
			opt['filter_gain'] = uniform_rand(1., 5.)
			opt['momentum'] = uniform_rand(0.85, 0.99)
			opt['psi_preconditioner'] = log_uniform_rand(1e-1, 1e1)
			opt['delay'] = int(uniform_rand(7,15))
	
			print
			for key, val in opt.iteritems():
				print(key + ': ' + str(val))
			print
			opt['y'] = run(opt)
		
		save_name = './logs/hyperopt_mean_pooling/numpy/trial'+str(trial_num)+'.pkl'
		with open(save_name, 'w') as fp:
			pkl.dump(opt, fp, protocol=pkl.HIGHEST_PROTOCOL)
		if opt['y'] > y_best:
			y_best = opt['y']
			best_params = opt
		
		print
		print
		print('Best y so far')
		print y_best
		print('Best params so far')	
		for key, val in best_params.iteritems():
			print('Best ' + key + ': ' + str(val))
		print
		print
		print
	
	print('Best y overall')
	print y_best
	print('Best params overall')	
	for key, val in best_params.iteritems():
		print('Best ' + key + ': ' + str(val))
	
	"""
	y = []
	best_params['combine_train_val'] = True
	for i in xrange(5):
		best_params['trial_num'] = 'T-'+str(i)
		y.append(run(best_params))
		print
		print('Current y: %f' % (y[i],))
		print
	
	print('Best y overall')
	print y_best
	print('Best params overall')	
	for key, val in opt.iteritems():
		print('Best ' + key + ': ' + str(val))
	# Compute statistics
	print y
	y = np.asarray(y)
	save_name = './logs/hyperopt_mean_pooling/numpy/test.npz'
	np.savez(save_name, y=y)
	
	mean = np.mean(y)
	print('Mean: %f' % (mean,))
	print('Std: %f' % (np.std(y),))
	"""

def uniform_rand(min_, max_):
	gap = max_ - min_
	return gap*np.random.rand() + min_

def log_uniform_rand(min_, max_, size=1):
	if size > 1:
		output = []
		for i in xrange(size):
			output.append(10**uniform_rand(np.log10(min_), np.log10(max_)))
	else:
		output = 10**uniform_rand(np.log10(min_), np.log10(max_))
	return output



if __name__ == '__main__':
	random_independent(10)
