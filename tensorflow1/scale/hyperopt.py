'''Hyperopt'''

import os
import sys
import time

import cPickle as pkl
import numpy as np
import tensorflow as tf

from rot_mnist import main


def run(n_trials):
	opt = {}
	opt['root'] = '/home/dworrall'
	dir_ = opt['root'] + '/Code/harmonicConvolutions/tensorflow1/scale'
	opt['n_epochs'] = 1000
	opt['im_size'] = (28,28)
	opt['train_size'] = 10000
	opt['loss_type_image'] = 'l2'
	opt['save_step'] = 10
	opt['mb_size'] = 128
	opt['lr_schedule'] = [550,750]
	# I think these are good
	opt['n_layers'] = 8
	
	best_test_acc = 0.
	best_opt = None
	test_accs = []
	options = []
	for i in xrange(n_trials):
		NaN = True
		while NaN:			
			
			opt['lr'] = np.power(10.,-3. + 0.2*np.random.rand())
			opt['equivariant_weight'] = 0.6 + 0.25*(2.*(np.random.rand()-0.5))
			opt['n_mid'] = int(20 + 2*np.random.randint(3))
			opt['n_mid_class'] = opt['n_mid']
			opt['n_layers_deconv'] = int(3 + 2*np.random.randint(2))
			opt['n_layers_class'] = int(2 + 2*np.random.randint(2))
			opt['dropout'] = 0.25*np.random.rand()
			
			opt['summary_path'] = dir_ + 'hyperopt/invariant/summaries/conv_rot_mnist_{:.0e}_{:s}'.format(opt['equivariant_weight'], 'hp')
			opt['save_path'] = dir_ + 'hyperopt/invariant/checkpoints/conv_rot_mnist_{:.0e}_{:s}/model.ckpt'.format(opt['equivariant_weight'], 'hp')
			options.append(opt)
			
			# Print options at beginning
			for key, val in opt.iteritems():
				print key, val
			
			# Run
			tf.reset_default_graph()
			test_acc = main(opt)
			NaN = test_acc < 0
		
		# Save
		test_accs.append(test_acc)
		np.save('./hyperopt/invariant/test_accs3.npy', test_acc)
		with open('./hyperopt/invariant/test_options3.pkl','w') as fp:
			pkl.dump(options, fp, protocol=pkl.HIGHEST_PROTOCOL)
		
		# Best
		if test_acc > best_test_acc:
			best_test_acc = test_acc
			best_opt = opt
		print
		print
		print('Best test acc: {:03f}'.format(best_test_acc))
		for key, val in best_opt.iteritems():
			print key, val
		print
		print


if __name__ == '__main__':
	run(35)
