import sys
import os
import numpy as np
import tensorflow as tf

from model_assembly_train import build_all_and_train
from settings import settings

if __name__ == '__main__':
	if len(sys.argv) < 5:
		print('Please provide at least:')
		print('     -comma-separated list of device IDxs to use')
		print('     -dataset name (rotated_mnist / cifar10)')
		print('     -model name (as defined in harmonic_network_models.py)')
		print('     -parent data directory')
		print('     -fraction of training data to use (will be ignored without using queues)')
		sys.exit(1)
	deviceIdxs = [int(x.strip()) for x in sys.argv[1].split(',')]
	opt = {}
	opt['deviceIdxs'] = deviceIdxs
	opt['dataset'] = sys.argv[2]
	opt['model'] = sys.argv[3]
	opt['data_dir'] = sys.argv[4]
	if len(sys.argv) >= 6:
		opt['train_data_fraction'] = float(sys.argv[5])
	
	#create configuration for different tests
	options = settings(opt)
	options.create_options()

	#build the model and train it
	build_all_and_train(options)
	print("ALL FINISHED! :)")