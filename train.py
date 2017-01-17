import sys
import os
import numpy as np
import tensorflow as tf

from model_assembly_train import build_all_and_train
from io_helpers import load_dataset, download_dataset, get_num_items_in_tfrecords
import harmonic_network_models

def create_opt_data(opt):
	# Default configuration
	opt['model'] = getattr(harmonic_network_models, opt['model'])
	opt['save_step'] = 10
	opt['display_step'] = 1e6
	opt['lr'] = 3e-2
	opt['batch_size'] = 50
	opt['n_epochs'] = 100
	opt['n_filters'] = 8
	opt['trial_num'] = 'A'
	opt['combine_train_val'] = False
	opt['std_mult'] = 0.3
	opt['filter_gain'] = 2
	opt['momentum'] = 0.93
	opt['psi_preconditioner'] = 3.4
	opt['delay'] = 8
	opt['lr_div'] = 10.
	opt['augment'] = False
	opt['crop_shape'] = 0
	opt['log_path'] = 'logs/current'
	opt['checkpoint_path'] = 'checkpoints/current'
	opt['is_bsd'] = False
	# Model specifics
	if opt['datasetIdx'] == 'mnist':
		# Download MNIST if it doesn't exist
		if not os.path.exists(opt['data_dir'] + '/mnist_rotation_new'):
			download_dataset(opt)
		# Load dataset
		mnist_dir = opt['data_dir'] + '/mnist_rotation_new'
		data = {}
		#data feeding choice
		opt['use_io_queues'] = False
		if opt['use_io_queues']:
			data['train_files'] = [mnist_dir + '/train.tfrecords']
			data['valid_files'] = [mnist_dir + '/valid.tfrecords']
			data['test_files'] = [mnist_dir + '/test.tfrecords']
			#get the number of items of each set
			data['train_items'] = get_num_items_in_tfrecords(data['train_files'])
			data['valid_items'] = get_num_items_in_tfrecords(data['valid_files'])
			data['test_items'] = get_num_items_in_tfrecords(data['test_files'])
			#let's define some functions to reshape data
			#note: [] means nothing will happen
			data['x_shape_target'] = [28, 28, 1]
			data['y_shape_target'] = [1]
		else:
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
		opt['aug_crop'] = 0 #'crop margin'
		opt['n_epochs'] = 200
		opt['batch_size'] = 46
		opt['lr']  = 0.0076
		opt['optimizer'] = tf.train.AdamOptimizer
		opt['momentum'] = 0.93
		opt['std_mult'] = 0.7
		opt['delay'] = 12
		opt['psi_preconditioner'] = 7.8
		opt['filter_gain'] = 2
		opt['filter_size'] = 3	# <<<< this should be 5!!!!!!!!!!!!!!!!
		opt['n_filters'] = 8
		opt['display_step'] = 10000/(opt['batch_size']*3.)
		opt['is_classification'] = True
		opt['combine_train_val'] = False
		opt['dim'] = 28
		opt['crop_shape'] = 0
		opt['n_channels'] = 1
		opt['n_classes'] = 10
		opt['log_path'] = './logs/deep_mnist/trialA'
		opt['checkpoint_path'] = './checkpoints/deep_mnist/trialA'
	elif opt['datasetIdx'] == 'cifar10':
		print("""WARNING: Our Deep CIFAR Model is new an experimental,
so the current version will be unstable unless otherwise noted!""") 
		# Download CIFAR10 if it doesn't exist
		if not os.path.exists(opt['data_dir'] + '/cifar_numpy'):
			download_dataset(opt)
		# Load dataset
		data = load_dataset(opt['data_dir'], 'cifar_numpy')
		opt['is_classification'] = True
		opt['dim'] = 32
		opt['crop_shape'] = 0
		opt['aug_crop'] = 3
		opt['n_channels'] = 3
		opt['n_classes'] = 10 
		opt['n_epochs'] = 250
		opt['batch_size'] = 32
		opt['lr']  = 0.01
		opt['optimizer'] = tf.train.AdamOptimizer
		opt['std_mult'] = 0.4
		opt['delay'] = 8
		opt['psi_preconditioner'] = 7.8
		opt['filter_gain'] = 2
		opt['filter_size'] = 3
		opt['n_filters'] = 4*10	# Wide ResNet
		opt['resnet_block_multiplicity'] = 3
		opt['augment'] = True
		opt['momentum'] = 0.93
		opt['display_step'] = 25
		opt['is_classification'] = True
		opt['n_channels'] = 3
		opt['n_classes'] = 10
		opt['log_path'] = './logs/deep_cifar'
		opt['checkpoint_path'] = './checkpoints/deep_cifar'
	elif opt['datasetIdx'] == 'plankton': 
		# Load dataset
		data = load_dataset(opt['data_dir'], 'plankton_numpy')
		data['train_x'] = np.squeeze(data['train_x'])
		data['valid_x'] = np.squeeze(data['valid_x'])
		opt['lr'] = 1e-1
		opt['batch_size'] = 32
		opt['std_mult'] = 1
		opt['momentum'] = 0.95
		opt['psi_preconditioner'] = 3.4
		opt['delay'] = 8
		opt['display_step'] = 10
		opt['save_step'] = 1
		opt['is_classification'] = True
		opt['n_epochs'] = 250
		opt['dim'] = 95
		opt['n_channels'] = 1
		opt['n_classes'] = 121
		opt['n_filters'] = 32
		opt['filter_gain'] = 2
		opt['augment'] = True
		opt['lr_div'] = 10.
		opt['trial_num'] = 'B'
		opt['crop_shape'] = 10
		opt['log_path'] = './logs/deep_plankton'
		opt['checkpoint_path'] = './checkpoints/deep_plankton'
	elif opt['datasetIdx'] == 'galaxies': 
		# Load dataset
		data = load_dataset(opt['data_dir'], 'galaxies_numpy')
		opt['is_classification'] = False
		opt['dim'] = 64
		opt['n_channels'] = 3
		opt['n_classes'] = 37
	elif opt['datasetIdx'] == 'bsd':
		opt['trial_num'] = 'R'
		opt['combine_train_val'] = False	
		data = load_pkl(opt['data_dir'], 'bsd_pkl_float', prepend='')
		opt['model'] = getattr(equivariant, 'deep_bsd')
		opt['is_bsd'] = True
		opt['lr'] = 1e-2
		opt['batch_size'] = 10
		opt['std_mult'] = 1
		opt['momentum'] = 0.95
		opt['psi_preconditioner'] = 3.4
		opt['delay'] = 8
		opt['display_step'] = 8
		opt['save_step'] = 10
		opt['is_classification'] = True
		opt['n_epochs'] = 250
		opt['dim'] = 321
		opt['dim2'] = 481
		opt['n_channels'] = 3
		opt['n_classes'] = 10
		opt['n_filters'] = 8 #32
		opt['filter_gain'] = 2
		opt['augment'] = True
		opt['lr_div'] = 10.
		opt['log_path'] = './logs/deep_bsd'
		opt['checkpoint_path'] = './checkpoints/deep_bsd'
		opt['test_path'] = './bsd/trial' + opt['trial_num']
		opt['anneal_sl'] = True
		opt['load_pretrained'] = False
		opt['sparsity'] = 1
		if not os.path.exists(opt['test_path']):
			os.mkdir(opt['test_path'])
		opt['save_test_step'] = 5
	else:
		print('Dataset unrecognized, options are:')
		print('mnist, cifar10, plankton, galaxies, bsd')
		sys.exit(1)
	return opt, data

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print('Please provide:')
		print('     -comma-separated list of device IDxs to use')
		print('     -dataset name (mnist / cifar10 /  plankton / galaxies / bsd)')
		print('     -model name (as defined in harmonic_network_models.py)')
		print('     -parent data directory')
		sys.exit(1)
	deviceIdxs = [int(x.strip()) for x in sys.argv[1].split(',')]
	opt = {}
	opt['deviceIdxs'] = deviceIdxs
	opt['datasetIdx'] = sys.argv[2]
	opt['model'] = sys.argv[3]
	opt['data_dir'] = sys.argv[4]
	#create configuration for different tests
	amended_opt, data = create_opt_data(opt)
	#build the model and train it
	build_all_and_train(amended_opt, data)
	print("ALL FINISHED! :)")