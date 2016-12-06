import sys
from model_assembly_train import build_all_and_train

def create_opt_data(opt):
	# Default configuration
	opt['data_dir'] = '/home/daniel/data'
	opt['model'] = getattr(equivariant, opt['model'])
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
	opt['log_path'] = './logs/current'
	opt['checkpoint_path'] = './checkpoints/current'
	opt['is_bsd'] = False
	
	# Model specifics
	if opt['datasetIdx'] == 'mnist':
		# Load dataset
		mnist_dir = opt['data_dir'] + '/mnist_rotation_new'
		train = np.load(mnist_dir + '/rotated_train.npz')
		valid = np.load(mnist_dir + '/rotated_valid.npz')
		test = np.load(mnist_dir + '/rotated_test.npz')
		data = {}
		#dataset choice
		data['train_x'] = train['x']
		data['train_y'] = train['y']
		data['valid_x'] = valid['x']
		data['valid_y'] = valid['y']
		data['test_x'] = test['x']
		data['test_y'] = test['y']
		opt['n_epochs'] = 200
		opt['batch_size'] = 46
		opt['lr']  = 0.0076
		opt['momentum'] = 0.93
		opt['std_mult'] = 0.7
		opt['delay'] = 12
		opt['psi_preconditioner'] = 7.8
		opt['filter_gain'] = 2
		opt['filter_size'] = 3
		opt['n_filters'] = 8
		opt['display_step'] = 10000/(opt['batch_size']*3.)
		opt['is_classification'] = True
		opt['combine_train_val'] = True
		opt['dim'] = 28
		opt['crop_shape'] = 0
		opt['n_channels'] = 1
		opt['n_classes'] = 10
		opt['log_path'] = './logs/deep_mnist/trialA'
		opt['checkpoint_path'] = './checkpoints/deep_mnist/trialA'
	elif opt['datasetIdx'] == 'cifar10': 
		# Load dataset
		data = load_dataset(opt['data_dir'], 'cifar_numpy')
		opt['is_classification'] = True
		opt['dim'] = 32
		opt['crop_shape'] = 0
		opt['n_channels'] = 3
		opt['n_classes'] = 10 
		opt['n_epochs'] = 80
		opt['batch_size'] = 64
		opt['lr']  = 0.01
		opt['std_mult'] = 0.7
		opt['delay'] = 8
		opt['psi_preconditioner'] = 7.8
		opt['filter_gain'] = 2.1
		opt['n_filters'] = 32
		opt['momentum'] = 0.93
		opt['display_step'] = 25
		opt['is_classification'] = True
		opt['dim'] = 32
		opt['crop_shape'] = 0
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
		data = load_dataset(opt['data_dir'], 'BSDS500_numpy', prepend='small_')
		data['train_y'] = data['train_y'][...,np.newaxis]
		data['valid_y'] = data['valid_y'][...,np.newaxis]
		del data['test_x']
		del data['test_y']
		opt['pos_weight'] = 100
		opt['model'] = getattr(equivariant, 'deep_bsd')
		opt['is_bsd'] = True
		opt['lr'] = 1e-1
		opt['batch_size'] = 4
		opt['std_mult'] = 1
		opt['momentum'] = 0.95
		opt['psi_preconditioner'] = 3.4
		opt['delay'] = 8
		opt['display_step'] = 10
		opt['save_step'] = 10
		opt['is_classification'] = True
		opt['n_epochs'] = 250
		opt['dim'] = 127
		opt['dim2'] = 160
		opt['n_channels'] = 3
		opt['n_classes'] = 2
		opt['n_filters'] = 32
		opt['filter_gain'] = 2
		opt['augment'] = False
		opt['lr_div'] = np.sqrt(10.)
		opt['crop_shape'] = 0
		opt['log_path'] = './logs/deep_bsd'
		opt['checkpoint_path'] = './checkpoints/deep_bsd'
	else:
		print('Dataset unrecognized, options are:')
		print('mnist, cifar10, plankton, galaxies, bsd')
		sys.exit(1)
		return opt, data

if __name__ == '__main__':
	deviceIdxs = [int(x.strip()) for x in sys.argv[2].split(',')]
	opt = {}
	opt['model'] = sys.argv[3]
	opt['datasetIdx'] = sys.argv[1]
	opt['deviceIdxs'] = deviceIdxs
	#create configuration for different tests
	opt, data = create_opt_data(opt)
    #build the model and train it
	build_all_and_train(opt, data)
	print("ALL FINISHED! :)")