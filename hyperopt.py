'''Hyperparameter optimizer'''

import os
import shutil
import sys
import time

import numpy as np

from trainModel import run
##### HELPERS #####
def checkFolder(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def random_independent(n_trials=3, datasetIdx=0, deviceIdxs=[0], model='deep_complex_bias'):
	y_best = 0.
	best_params = {}
	opt = {}
	datasetName = ''
	if datasetIdx == 0:
		datasetName = 'MNIST'
	elif datasetIdx == 1:
		datasetName = 'CIFAR10'
	elif datasetIdx == 2:
		datasetName = 'PLANKTON'
	elif datasetIdx == 3:
		datasetName = 'GALAXIES'
	else:
		datasetName = 'UNKOWN_DATASET'
	for i in xrange(n_trials):
		opt['y'] = -1
		while opt['y'] < 0:
			# Remove files corresponding to trial
			trial_num = i + 10
			log_path = './logs/hyperopt_stable/trial'+str(trial_num)
			log_path = log_path + '_' + datasetName
			if os.path.exists(log_path):
				shutil.rmtree(log_path)
			checkpoint_path = './checkpoints/hyperopt_stable/trial'+str(trial_num)
			checkpoint_path = checkpoint_path + '_' + datasetName
			if os.path.exists(checkpoint_path):
				shutil.rmtree(checkpoint_path)

			opt['model'] = model
			opt['lr'] = log_uniform_rand(5e-3, 1e-1)
			opt['batch_size'] = int(log_uniform_rand(40,120))
			opt['n_epochs'] = 80
			opt['n_filters'] = int(uniform_rand(6,10))
			opt['trial_num'] = trial_num
			opt['combine_train_val'] = False
			opt['std_mult'] = uniform_rand(0.1, 1.)
			opt['filter_gain'] = uniform_rand(2., 4.)
			opt['momentum'] = uniform_rand(0.9, 0.99)
			opt['psi_preconditioner'] = log_uniform_rand(1e-1, 1e1)
			opt['delay'] = int(uniform_rand(7,15))
			opt['datasetIdx'] = datasetIdx
			opt['deviceIdxs'] = deviceIdxs
			opt['displayStep'] = 10
			opt['augment'] = False
			opt['log_path'] = log_path
			opt['checkpoint_path'] = checkpoint_path
			print
			for key, val in opt.iteritems():
				print(key + ': ' + str(val))
			print
			opt['y'] = run(opt)
		
		save_name = './logs/hyperopt_stable/numpy/trial'+str(trial_num)+'.pkl'
		with open(save_name, 'w') as fp:
			pkl.dump(opt, fp, protocol=pkl.HIGHEST_PROTOCOL)
		if opt['y'] > y_best:
			y_best = opt['y']
			best_params = opt.copy()
		
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
	print("datasetIdx: ", int(sys.argv[1]))
	deviceIdxs = [int(x.strip()) for x in sys.argv[2].split(',')]
	print("deviceIdxs : ", deviceIdxs)
	print("NetworkModel : ", sys.argv[3])
	random_independent(n_trials=24, datasetIdx=int(sys.argv[1]), deviceIdxs=deviceIdxs, model=sys.argv[3]) #SWITCH MNIST/CIFAR
	print("ALL FINISHED! :)")
