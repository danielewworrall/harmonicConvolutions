import os
import sys
import time

import numpy as np

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

def random_independent(n_trials=3):
    learning_rates = np.zeros(n_trials)
    batch_sizes = np.zeros(n_trials)
    stddev_multipliers = np.zeros(n_trials)

    for i in xrange(n_trials):
        n_epochs = 500
        n_filters = 10

        lr = log_uniform_rand(1e-2, 1e-4)
        batch_size = int(log_uniform_rand(64,256))
        std_mult = uniform_rand(0.05, 1.0)
        print
        print('Learning rate: %f' % (lr,))
        print('Batch size: %f' % (batch_size,))
        print('Stddev multiplier: %f' % (std_mult,))
        #remember this random combination
        learning_rates[i] = lr
        batch_sizes[i] = batch_size
        stddev_multipliers[i] = std_mult

    #save these values so we can comparse test scores
    np.save("trialParams/learning_rates.npy", learning_rates)
    np.save("trialParams/batch_sizes.npy", batch_sizes)
    np.save("trialParams/stddev_multipliers.npy", stddev_multipliers)

random_independent(30)