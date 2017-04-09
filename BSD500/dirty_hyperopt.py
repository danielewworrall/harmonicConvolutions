'''Dirty hyperopt'''

import os
import sys
import time

import numpy as np

from run_BSD import main

"""Run the model for about 10 epochs and choose the one with the
hyperparameters, which minimize the objective the fastest
"""

def looper(n_trials):
	opt = {}
	opt['combine_train_val'] = False	
	#data = load_pkl('./data', 'bsd_pkl_float', prepend='')
	opt['test_path'] = 'my_model'
	opt['log_path'] = './logs/' + opt['test_path']
	opt['checkpoint_path'] = './checkpoints/' + opt['test_path']
	opt['test_path'] = './' + opt['test_path']
	
	opt['sparsity'] = 1.
	opt['augment'] = True
	opt['filter_gain'] = 2
	opt['dim'] = 321
	opt['dim2'] = 481
	opt['n_channels'] = 3
	opt['n_classes'] = 10
	opt['display_step'] = 8
	opt['save_step'] = 11
	opt['n_epochs'] = 10
	opt['delay'] = 8
	opt['lr_div'] = 10.
	
	opt['load_settings'] = False
	
	best = 1e6
	best_opt = {}
	for i in xrange(n_trials):
		opt['learning_rate'] = log_rand(1e-2, 1e-1)
		
		opt['batch_size'] = int(rand(5,10))
		opt['std_mult'] = rand(0.2, 0.5)
		opt['psi_preconditioner'] = rand(2.,5.)

		opt['n_filters'] = 7
		opt['filter_size'] = 3
		opt['n_rings'] = int(rand(1,4))
		train_loss, n_vars = main(opt)
		opt['n_vars'] = n_vars
		
		if train_loss < best:
			best_opt = opt
			best = np.minimum(best, train_loss)
		
		print
		print('Best train_loss so far: {:f}'.format(best))
		for key, val in best_opt.iteritems():
			print(key, val)
		print
	
	print('Best train_loss: {:f}'.format(best))
	for key, val in best_opt.iteritems():
		print(key, val)


def rand(low, high):
	return low + (high-low)*np.random.rand()


def log_rand(low, high):
	log_low = np.log10(low)
	log_high = np.log10(high)
	return np.power(10., log_low + (log_high-log_low)*np.random.rand())


if __name__ == '__main__':
	n_trials = 16
	looper(n_trials)
