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
	opt['n_epochs'] = 1500
	opt['im_size'] = (28,28)
	opt['train_size'] = 10000
	opt['loss_type_image'] = 'l2'
	opt['save_step'] = 10
	
	best_test_acc = 0.
	best_opt = None
	test_accs = []
	options = []
	for i in xrange(n_trials):
		NaN = True
		while NaN:
			opt['mb_size'] = int(np.random.randint(100)+150)
			opt['lr_schedule'] = [600+int(np.random.randint(100)), 1000+int(np.random.randint(250))]
			opt['lr'] = np.power(10.,4*np.random.rand()-6)
			opt['equivariant_weight'] = 0.65*(np.random.rand()-0.5) + 0.5
			opt['n_mid'] = int(10 + 2*np.random.randint(5))
			opt['n_layers'] = int(2 + np.random.randint(3))
			opt['n_mid_class'] = opt['n_mid']
			opt['n_layers_class'] = int(4 + np.random.randint(2))
			
			opt['summary_path'] = dir_ + 'hyperopt/narrow/summaries/conv_rot_mnist_{:.0e}_{:s}'.format(opt['equivariant_weight'], 'hp')
			opt['save_path'] = dir_ + 'hyperopt/narrow/checkpoints/conv_rot_mnist_{:.0e}_{:s}/model.ckpt'.format(opt['equivariant_weight'], 'hp')
			options.append(opt)
			
			# Run
			tf.reset_default_graph()
			test_acc = main(opt)
			NaN = test_acc < 0
		
		# Save
		test_accs.append(test_acc)
		np.save('./hyperopt/narrow/test_accs1.npy', test_acc)
		with open('./hyperopt/narrow/test_options1.pkl','w') as fp:
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
	run(25)
