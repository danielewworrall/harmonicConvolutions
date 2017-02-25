'''Ablation study'''

import os
import sys
import time

import numpy

from trainModel import *

def subsample_data(x,y,frac):
	"""Take first frac of data"""
	n_data = x.shape[0]
	n_samples = int(n_data*frac)
	return x[:n_samples,:], y[:n_samples]

def run(frac, model):
	# Parameters
	tf.reset_default_graph()
	
	opt = {}
	# Default configuration
	opt['deviceIdxs'] = [0,]
	opt['data_dir'] = '/home/daniel/data'
	opt['model'] = getattr(equivariant, model)
	opt['save_step'] = 10
	opt['trial_num'] = 'O'
	opt['lr_div'] = 10.
	opt['augment'] = False
	opt['is_bsd'] = False
	opt['datasetIdx'] = 'mnist'
	
	# Load dataset
	mnist_dir = opt['data_dir'] + '/mnist_rotation_new'
	train = np.load(mnist_dir + '/rotated_train.npz')
	valid = np.load(mnist_dir + '/rotated_valid.npz')
	test = np.load(mnist_dir + '/rotated_test.npz')
	data = {}
	data['train_x'] = train['x']
	data['train_y'] = train['y']
	data['valid_x'] = valid['x']
	data['valid_y'] = valid['y']
	data['test_x'] = test['x']
	data['test_y'] = test['y']
	data['train_x'], data['train_y'] = subsample_data(data['train_x'], data['train_y'], frac)
	data['valid_x'], data['valid_y'] = subsample_data(data['valid_x'], data['valid_y'], frac)
	opt['n_epochs'] = 200
	opt['batch_size'] = 46
	opt['lr']  = 0.0076
	opt['std_mult'] = 0.7
	opt['delay'] = 12
	opt['psi_preconditioner'] = 7.8
	opt['filter_gain'] = 1.
	opt['filter_size'] = 3
	opt['n_filters'] = 20
	opt['momentum'] = 0.93
	opt['display_step'] = 10000/(opt['batch_size']*3.)
	opt['is_classification'] = True
	opt['combine_train_val'] = True
	opt['dim'] = 28
	opt['crop_shape'] = 0
	opt['n_channels'] = 1
	opt['n_classes'] = 10
	opt['log_path'] = './logs/deep_mnist/trialO'
	opt['checkpoint_path'] = './checkpoints/deep_mnist/trialO'
	
	# Check that save paths exist
	opt['log_path'] = opt['log_path'] + '/trial' + str(opt['trial_num'])
	opt['checkpoint_path'] = opt['checkpoint_path'] + '/trial' + \
							str(opt['trial_num']) 
	if not os.path.exists(opt['log_path']):
		print('Creating log path')
		os.mkdir(opt['log_path'])
	if not os.path.exists(opt['checkpoint_path']):
		print('Creating checkpoint path')
		os.mkdir(opt['checkpoint_path'])
	opt['checkpoint_path'] = opt['checkpoint_path'] + '/model.ckpt'
	
	# Print out options
	for key, val in opt.iteritems():
		print(key + ': ' + str(val))
	return train_model(opt, data)


def ablation_runs(n=5):
	"""Run n runs over varying sizes of the data"""
	model = 'deep_Z'
	acc = []
	fracs = []
	for frac in np.linspace(1./n,1.,n):
		print frac
		acc.append(run(frac, model))
		fracs.append(frac)
		print acc
		np.savez('./ablation_O', acc=acc, fracs=fracs)
	

if __name__ == '__main__':
	ablation_runs(6)