'''Hyperparameter optimizer'''

import os
import sys
import time

import numpy as np

from equivariant import run

def random_independent(n_trials=3):
	y_best = 0.
	best_params = {}
	for i in xrange(n_trials):
		n_epochs = 500
		n_filters = 10
		
		lr = log_uniform_rand(5e-3, 1e-1)
		lr_decay = log_uniform_rand(1e-3, 1e-1)
		batch_size = int(log_uniform_rand(80,120))
		std_mult = uniform_rand(0.05, 0.2)
		print
		print('Learning rate: %f' % (lr,))
		print('Learning rate decay: %f' % (lr_decay,))
		print('Batch size: %f' % (batch_size,))
		print('Stddev multiplier: %f' % (std_mult,))
		print
		y = run(model='conv_so2',
				lr=lr,
				lr_decay=lr_decay,
				batch_size=batch_size,
				std_mult=std_mult,
				n_epochs=n_epochs,
				n_filters=n_filters,
				trial_num=i,
				combine_train_val=False)
		save_name = './hyperopt_logs_nesterov/trial'+str(i)+'.npz'
		np.savez(save_name, y=y, lr=lr, lr_decay=lr_decay,
				 batch_size=batch_size, std_mult=std_mult)
		if y > y_best:
			y_best = y
			best_params['lr'] = lr
			best_params['lr_decay'] = lr_decay
			best_params['batch_size'] = batch_size
			best_params['std_mult'] = std_mult
		
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
		y.append(run(model='conv_so2',
					 lr=best_params['lr'],
					 lr_decay=best_params['lr_decay'],
					 batch_size=best_params['batch_size'],
					 std_mult=best_params['std_mult'],
					 n_epochs=n_epochs,
					 n_filters=n_filters,
					 trial_num='T-'+str(i),
					 combine_train_val=True))
		print
		print('Current y: %f' % (y[i],))
		print
	
	print('Best y overall')
	print y_best
	print('Best params overall')	
	print best_params
	# Compute statistics
	print y
	y = np.asarray(y)
	save_name = './hyperopt_logs/test.npz'
	np.savez(save_name, y=y)
	
	mean = np.mean(y)
	print('Mean: %f' % (mean,))
	print('Std: %f' % (np.std(y),))

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
	random_independent(24)
