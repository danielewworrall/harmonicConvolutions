import sys
import os
import numpy as np
import tensorflow as tf

from model_assembly_train import build_all_and_train
from settings import settings

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print('Please provide:')
		print('     -comma-separated list of device IDxs to use')
		print('     -dataset name (rotated_mnist / cifar10)')
		print('     -model name (as defined in harmonic_network_models.py)')
		print('     -parent data directory')
		sys.exit(1)
	deviceIdxs = [int(x.strip()) for x in sys.argv[1].split(',')]
	opt = {}
	opt['deviceIdxs'] = deviceIdxs
	opt['dataset'] = sys.argv[2]
	opt['model'] = sys.argv[3]
	opt['data_dir'] = sys.argv[4]
	opt['train_data_fraction'] = 0.1

	#create configuration for different tests
	options = settings(opt)
	options.create_options()

	#build the model and train it
	build_all_and_train(options)
	print("ALL FINISHED! :)")