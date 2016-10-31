'''Hyperparameter optimizer'''

import os
import sys
import time

import numpy as np

from trainModel import run
##### HELPERS #####
def checkFolder(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


def random_independent(n_trials=3, fixedParams = True, experimentIdx = 0, deviceIdxs=[0], model=''):
	y_best = 0.
	best_params = {}
	best_num_filters = 0
	actual_n_trials = n_trials
	y_s = [] #all results
	if fixedParams:
		learning_rates = np.load("trialParams/learning_rates.npy")
		batch_sizes = np.load("trialParams/batch_sizes.npy")
		stddev_multipliers = np.load("trialParams/stddev_multipliers.npy")
		#for debug
		print("learning rates:", learning_rates)
		print("batch sizes:", batch_sizes)
		print("stddevs:", stddev_multipliers)
		#make sure user parameters agree
		if n_trials != learning_rates.shape[0]:
			print("WARNING: Setting ntrials to loaded experiment files: ", learning_rates.shape[0])
			actual_n_trials = learning_rates.shape[0]
	#number of filters to try
	filters = [8]
	print("Num trials per filter", actual_n_trials)
	for f in filters:
		local_y_s = []
		print("Processsing for num Filters:", f)
		for i in xrange(actual_n_trials):
			n_epochs = 500

			#switch here as well
			if fixedParams:
				lr = learning_rates[i]
				batch_size = int(batch_sizes[i])
				std_mult = stddev_multipliers[i]
			else:
				lr = log_uniform_rand(1e-2, 1e-4)
				batch_size = int(log_uniform_rand(64,256))
				std_mult = uniform_rand(0.05, 1.0)
			if batch_size % len(deviceIdxs) != 0:
				while batch_size % len(deviceIdxs) != 0:
					batch_size += 1
				batch_size = batch_size + 1
				print("WARNING: Setting batch size to be divisible by number of GPUs.")
			print
			print('Learning rate: %f' % (lr,))
			print('Batch size: %f' % (batch_size,))
			print('Stddev multiplier: %f' % (std_mult,))
			print
			y = run(model=model,
				lr=lr,
				batch_size=batch_size,
				std_mult=std_mult,
				n_epochs=n_epochs,
				n_filters=f,
				trial_num=i,
				combine_train_val=False,
				experimentIdx = experimentIdx,
				deviceIdxs = deviceIdxs,
				use_batchNorm = True)
			local_y_s.append(y)
			if y > y_best:
				y_best = y
				best_params['lr'] = lr
				best_params['batch_size'] = batch_size
				best_params['std_mult'] = std_mult
				best_num_filters = f
		#remember all ys 
		y_s.append(local_y_s)
		y_s_a = np.asarray(y_s)
		#save temp
		if experimentIdx == 0:
				fileName = "results/MNIST/bestYResultsMNIST_temp_" + str(f) + ".npy"
		elif experimentIdx == 1:
				fileName = "results/CIFAR/bestYResultsCIFAR_temp_" + str(f) + ".npy"
		elif experimentIdx == 2:
				fileName = "results/PLANKTON/bestYResultsPLANKTON_temp_" + str(f) + ".npy"
		elif experimentIdx == 3:
				fileName = "results/GALAXIES/bestYResultsGALAXIES_temp_" + str(f) + ".npy"
		checkFolder(os.path.dirname(fileName))
		np.save(fileName, y_s_a)
		
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
	#save y_s
	if experimentIdx == 0:
		fileName = "results/yResultsMNIST.npy" 
	elif experimentIdx == 1:
		fileName = "results/yResultsCIFAR.npy" 
	np.save(fileName, np.asarray(y_s))

	y = []
	for i in xrange(5):
		y.append(run(model=model,
			lr=best_params['lr'],
			batch_size=best_params['batch_size'],
			std_mult=best_params['std_mult'],
			n_epochs=n_epochs,
			n_filters=best_num_filters,
			trial_num='T-'+str(i),
			combine_train_val=True,
			experimentIdx = experimentIdx,
			deviceIdxs = deviceIdxs,
			use_batchNorm = True))		
		print
		print('Current y: %f' % (y[i],))
		print

	print('Best num filters:', best_num_filters)
	print('Best y overall:')
	print y_best
	print('Best params overall:')	
	print best_params
	# Compute statistics
	print y
	y = np.asarray(y)
	mean = np.mean(y)
	print('Mean: %f' % (mean,))
	print('Std: %f' % (np.std(y),))
	#save y
	if experimentIdx == 0:
			fileName = "results/MNIST/bestYResultsMNIST.npy"
	elif experimentIdx == 1:
			fileName = "results/CIFAR/bestYResultsCIFAR.npy"
	elif experimentIdx == 2:
			fileName = "results/PLANKTON/bestYResultsPLANKTON.npy"
	elif experimentIdx == 3:
			fileName = "results/GALAXIES/bestYResultsGALAXIES_temp.npy"
	np.save(fileName, y)

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


#ENTRY POINT:
if __name__ == '__main__':
	print("experimentIdx: ", int(sys.argv[1]))
	deviceIdxs = [int(x.strip()) for x in sys.argv[2].split(',')]
	print("deviceIdxs : ", deviceIdxs)
	print("NetworkModel : ", sys.argv[3])
	random_independent(n_trials=24, fixedParams=True, experimentIdx=int(sys.argv[1]), deviceIdxs=deviceIdxs, model=sys.argv[3]) #SWITCH MNIST/CIFAR
	print("ALL FINISHED! :)")
