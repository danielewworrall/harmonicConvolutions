'''Multiple runs'''

import os
import sys
import time

import equivariant
import numpy as np
import tensorflow as tf

from trainModel import train_model

def multi_run(deviceIdx=0):
	opt = {}	
	opt['datasetIdx'] = 'mnist'
	opt['deviceIdxs'] = deviceIdx
	opt['model'] = getattr(equivariant, 'deep_stable')
	opt['n_epochs'] = 120
	opt['batch_size'] = 46
	opt['lr']  = 0.00756
	opt['lr_div'] = 10.
	opt['std_mult'] = 0.7
	opt['delay'] = 12
	opt['psi_preconditioner'] = 7.82
	opt['filter_gain'] = 2.1
	opt['n_filters'] = 8
	opt['momentum'] = 0.933
	opt['is_classification'] = True
	opt['dim'] = 28
	opt['crop_shape'] = 0
	opt['n_channels'] = 1
	opt['n_classes'] = 10
	opt['combine_train_val'] = True
	opt['save_step'] = 10
	opt['display_step'] = 1e6	
	opt['augment'] = False
	opt['crop_shape'] = 0
	log_path = './logs/deep_mnist'
	checkpoint_path = './checkpoints/deep_mnist'
	
	mnist_dir = '/home/sgarbin/data/mnist_rotation_new'
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
	
	# Print out options
	for key, val in opt.iteritems():
		print(key + ': ' + str(val))

	y = []
	for i in xrange(5):
		tf.reset_default_graph()
		opt['trial_num'] = i
		
		opt['log_path'] = log_path + '/trial' + str(opt['trial_num'])
		opt['checkpoint_path'] = checkpoint_path + '/trial' + str(opt['trial_num']) 
		if not os.path.exists(opt['log_path']):
			print('Creating log path')
			os.mkdir(opt['log_path'])
		if not os.path.exists(opt['checkpoint_path']):
			print('Creating checkpoint path')
			os.mkdir(opt['checkpoint_path'])
		opt['checkpoint_path'] = opt['checkpoint_path'] + '/model.ckpt'
		
		y.append(train_model(opt, data))
		np.save(opt['log_path'] + '/multi_run.npy', y)
		print
		print('Mean: %f' % (np.mean(y),))
		print('Standard deviation: %f' % (np.std(y),))
		print
	
	print('Best y overall')
	print y_best
	print('Best params overall')	
	for key, val in best_params.iteritems():
		print('Best ' + key + ': ' + str(val))


if __name__ == '__main__':
	deviceIdxs = [int(x.strip()) for x in sys.argv[1].split(',')]
	multi_run(deviceIdxs)
