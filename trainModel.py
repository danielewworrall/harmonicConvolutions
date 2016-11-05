import os
import sys
import time

import cv2
import equivariant
import numpy as np
import scipy as sp
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import tensorflow as tf

import input_data

from equivariant import *
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import misc
from steer_conv import *

###HELPER FUNCTIONS------------------------------------------------------------------
def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.
	Note that this function provides a synchronization point across all towers.

	tower_grads: List of lists of (gradient, variable) tuples. The outer list
	is over individual gradients. The inner list is over the gradient
	calculation for each tower.
	
	Returns:
	List of pairs of (gradient, variable) where the gradient has been averaged
	across all towers.
	"""
	if len(tower_grads) == 1:
		return tower_grads[0]
	else:
		print('Processing %d sets of gradients.' % len(tower_grads))
	average_grads = []
	for grad_and_vars in zip(*tower_grads): #for each grad, vars set
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, v in grad_and_vars:
			if g == None: #if no gradient, don't average'
				continue
			expanded_g = tf.expand_dims(g, 0)
			grads.append(expanded_g)

		#concat only if we have any entries
		if len(grads) == 0:
			continue
		# Average over the 'tower' dimension.
		grad = tf.concat(0, grads)
		grad = tf.reduce_mean(grad, 0)
		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads

def get_loss(opt, pred, y):
	"""Return loss function for classification/regression"""
	if opt['is_classification']:
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
	else:
		cost = 0.5*tf.reduce_mean(tf.pow(y - pred, 2))
	print('  Constructed loss')
	return cost

def get_io_placeholders(opt):
	"""Return placeholders for classification/regression"""
	size = opt['dim'] - 2*opt['crop_shape']
	n_input = size*size*opt['n_channels']
	io_x = tf.placeholder(tf.float32, [opt['batch_size'], n_input], name='x')
	if opt['is_classification']:
		io_y = tf.placeholder(tf.int64, [opt['batch_size']], name='y')
	else:
		io_y = tf.placeholder(tf.float32, [opt['batch_size'],
											  opt['num_classes']], name='y')
	return io_x, io_y

def build_optimizer(cost, lr, opt):
	"""Apply the psi_precponditioner"""
	mmtm = tf.train.MomentumOptimizer
	optim = mmtm(learning_rate=lr, momentum=opt['momentum'], use_nesterov=True)
	
	grads_and_vars = optim.compute_gradients(cost)
	modified_gvs = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			g = opt['psi_preconditioner']*g
		modified_gvs.append((g, v))
	optimizer = optim.apply_gradients(modified_gvs)
	print('  Optimizer built')
	return optimizer

def get_evaluation(pred, y, opt):
	if opt['is_classification']:
		correct_pred = tf.equal(tf.argmax(pred, 1), y)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	else:
		accuracy = cost
	return accuracy

def build_feed_dict(opt, io, batch, lr, pt, lr_, pt_):
	'''Build a feed_dict appropriate to training regime'''
	batch_x, batch_y = batch
	fd = {lr : lr_, pt : pt_}
	bs = opt['batch_size']
	for g in xrange(len(opt['deviceIdxs'])):
		fd[io['x'][g]] = batch_x[g*bs:(g+1)*bs,:]
		fd[io['y'][g]] = batch_y[g*bs:(g+1)*bs]
	return fd

##### TRAINING LOOPS #####
def loop(mode, sess, io, opt, data, cost, acc, lr, lr_, pt, optim=None, step=0):
	"""Run a loop"""
	X = data[mode+'_x']
	Y = data[mode+'_y']
	is_training = (mode=='train')
	n_GPUs = len(opt['deviceIdxs'])
	generator = minibatcher(X, Y, n_GPUs*opt['batch_size'],
							shuffle=is_training, augment=opt['augment'],
							img_shape=(opt['dim'], opt['dim']),
							crop_shape=opt['crop_shape'])
	cost_total = 0.
	acc_total = 0.
	for i, batch in enumerate(generator):
		fd = build_feed_dict(opt, io, batch, lr, pt, lr_, is_training)
		if mode == 'train':
			__, cost_, acc_ = sess.run([optim, cost, acc], feed_dict=fd)
		else:
			cost_, acc_ = sess.run([cost, acc], feed_dict=fd)
		if step % opt['display_step'] == 0:
			print('  ' + mode + ' Acc.: %f' % acc_)
		cost_total += cost_
		acc_total += acc_
		step += 1
	return cost_total/(i+1.), acc_total/(i+1.), step

def construct_model_and_optimizer(opt, io, lr, pt):
	"""Build the model and an single/multi-GPU optimizer"""
	if len(opt['deviceIdxs']) == 1:
		pred = opt['model'](opt, io['x'][0], opt['batch_size'], pt)
		loss = get_loss(opt, pred, io['y'][0])
		accuracy = get_evaluation(pred, io['y'][0], opt)
		train_op = build_optimizer(loss, lr, opt)
	else:
		# Multi_GPU Optimizer
		mmtm = tf.train.MomentumOptimizer
		optim = mmtm(learning_rate=lr, momentum=opt['momentum'], use_nesterov=True)
		#setup model for each GPU
		linearGPUIdx = 0
		gradientsPerGPU = []
		lossesPerGPU = []
		accuracyPerGPU = []
		for g in opt['deviceIdxs']:
			with tf.device('/gpu:%d' % g):
				print('Building Model on GPU: %d' % g)
				with tf.name_scope('%s_%d' % (opt['model'].__name__, 0)) as scope:
					# Forward pass
					pred = opt['model'](opt, io['x'][linearGPUIdx], pt)
					loss = get_loss(opt, pred, io['y'][linearGPUIdx])
					accuracy = get_evaluation(pred, io['y'][linearGPUIdx], opt)
					# Reuse variables for the next tower
					tf.get_variable_scope().reuse_variables()
					# Calculate gradients for minibatch on this tower
					grads = optim.compute_gradients(loss)
					# Keep track of gradients/losses/accuracies across all towers
					gradientsPerGPU.append(grads)
					lossesPerGPU.append(loss)
					accuracyPerGPU.append(accuracy)
			linearGPUIdx += 1
		# CPU-side synchronisation 
		grads = average_gradients(gradientsPerGPU)
		apply_gradient_op = optim.apply_gradients(grads)
		train_op = tf.group(apply_gradient_op)
	return loss, accuracy, train_op

def train_model(opt, data):
	"""Generalized training function
	
	opt: dict of options
	data: dict of numpy data
	"""
	n_GPUs = len(opt['deviceIdxs'])
	print('Using Multi-GPU Model with %d devices.' % n_GPUs)
	# Make placeholders
	io = {}
	io['x'] = []
	io['y'] = []
	for g in opt['deviceIdxs']:
		with tf.device('/gpu:%d' % g):
			io_x, io_y = get_io_placeholders(opt)
			io['x'].append(io_x)
			io['y'].append(io_y)
	lr = tf.placeholder(tf.float32, name='learning_rate')
	pt = tf.placeholder(tf.bool, name='phase_train')
	
	# Construct model and optimizer
	loss, accuracy, train_op = construct_model_and_optimizer(opt, io, lr, pt)
	
	# Initializing the variables
	init = tf.initialize_all_variables()
	if opt['combine_train_val']:
		data['train_x'] = np.vstack([data['train_x'], data['valid_x']])
		data['train_y'] = np.hstack([data['train_y'], data['valid_y']])

	# Summary writers
	tcost_ss = create_scalar_summary('training_cost')
	vcost_ss = create_scalar_summary('validation_cost')
	vacc_ss = create_scalar_summary('validation_accuracy')
	lr_ss = create_scalar_summary('learning_rate')

	# Configure tensorflow session
	config = config_init()
	if n_GPUs == 1:
		config.inter_op_parallelism_threads = 1 #prevent inter-session threads?
	sess = tf.Session(config=config)
	summary = tf.train.SummaryWriter(opt['log_path'], sess.graph)
	print('Summaries constructed...')
	
	sess.run(init)
	saver = tf.train.Saver()
	start = time.time()
	lr_ = opt['lr']
	epoch = 0
	step = 0.
	counter = 0
	best = 0.
	bs = opt['batch_size']
	print('Starting training loop...')
	while epoch < opt['n_epochs']:
		# Need batch_size*n_GPUs amount of data
		cost_total, acc_total, step = loop('train', sess, io, opt, data, loss,
										   accuracy, lr, lr_, pt, optim=train_op,
										   step=step)
		if not opt['combine_train_val']:
			vloss_total, vacc_total, __ = loop('valid', sess, io, opt, data,
											   loss, accuracy, lr, lr_,
											   pt, optim=train_op)

		fd = {tcost_ss[0] : cost_total, vcost_ss[0] : vloss_total,
			  vacc_ss[0] : vacc_total, lr_ss[0] : lr_}
		summaries = sess.run([tcost_ss[1], vcost_ss[1], vacc_ss[1], lr_ss[1]],
			feed_dict=fd)
		for summ in summaries:
			summary.add_summary(summ, step)

		best, counter, lr_ = get_learning_rate(opt, vacc_total, best, counter, lr_)

		print "[" + str(opt['trial_num']),str(epoch) + \
		"] Time: " + \
		"{:.3f}".format(time.time()-start) + ", Counter: " + \
		"{:d}".format(counter) + ", Loss: " + \
		"{:.5f}".format(cost_total) + ", Val loss: " + \
		"{:.5f}".format(vloss_total) + ", Train Acc: " + \
		"{:.5f}".format(acc_total) + ", Val acc: " + \
		"{:.5f}".format(vacc_total)
		epoch += 1

		if (epoch) % opt['save_step'] == 0:
			save_path = saver.save(sess, opt['checkpoint_path'])
			print("Model saved in file: %s" % save_path)
			
	if (opt['datasetIdx'] == 'plankton') or (opt['datasetIdx'] == 'galaxies'):
		tacc_total = vacc_total
	else:
		print('Testing')
		__, tacc_total, __ = loop('test', sess, io, opt, data, loss,
								  accuracy, lr, lr_, pt)
		print('Test accuracy: %f' % (tacc_total,))
	
	# Save model and exit
	save_path = saver.save(sess, opt['checkpoint_path'])
	print("Model saved in file: %s" % save_path)
	sess.close()
	return tacc_total

def load_dataset(dir_name, subdir_name):
	"""Load dataset from subdirectory"""
	data_dir = dir_name + '/' + subdir_name
	data = {}
	data['train_x'] = np.load(data_dir + '/trainX.npy')
	data['train_y'] = np.load(data_dir + '/trainY.npy')
	data['valid_x'] = np.load(data_dir + '/validX.npy')
	data['valid_y'] = np.load(data_dir + '/validY.npy')
	data['test_x'] = np.load(data_dir + '/testX.npy')
	if os.path.exists(data_dir + '/testY.npy'):
		data['test_y'] = np.load(data_dir + '/testY.npy')
	return data

def create_scalar_summary(name):
	"""Create a scalar summary placeholder and op"""
	ss = []
	ss.append(tf.placeholder(tf.float32, [], name=name))
	ss.append(tf.scalar_summary(name+'_summary', ss[0]))
	return ss

def config_init():
	"""Default config settings"""
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = False
	return config

##### MAIN SCRIPT #####
def run(opt):
	# Parameters
	data_dir = '/home/sgarbin/data'
	tf.reset_default_graph()
	
	# Default configuration
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
	opt['lr_div'] = np.sqrt(10.)
	opt['augment'] = False
	opt['crop_shape'] = 0
	opt['log_path'] = './logs/current'
	opt['checkpoint_path'] = './checkpoints/current'	
	
	# Model specifics
	if opt['datasetIdx'] == 'mnist':
		# Load dataset
		mnist_dir = data_dir + '/mnist_rotation_new'
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
		opt['lr']  = 3e-2
		opt['std_mult'] = 1.
		opt['display_step'] = 10000/(opt['batch_size']*3.)
		opt['is_classification'] = True
		opt['dim'] = 28
		opt['crop_shape'] = 0
		opt['n_channels'] = 1
		opt['n_classes'] = 10
		opt['log_path'] = './logs/deep_mnist'
		opt['checkpoint_path'] = './checkpoints/deep_mnist'
	elif opt['datasetIdx'] == 'cifar10': 
		# Load dataset
		data = load_dataset(data_dir, 'cifar_numpy')
		opt['is_classification'] = True
		opt['dim'] = 32
		opt['crop_shape'] = 0
		opt['n_channels'] = 3
		opt['n_classes'] = 10 
	elif opt['datasetIdx'] == 'plankton': 
		# Load dataset
		data = load_dataset(data_dir, 'plankton_numpy')
		opt['lr'] = 1.
		opt['batch_size'] = 32
		opt['std_mult'] = 1e-1
		opt['momentum'] = 0.95
		opt['psi_preconditioner'] = 3.4
		opt['delay'] = 8
		opt['display_step'] = 10
		opt['is_classification'] = True
		opt['n_epochs'] = 150
		opt['dim'] = 95
		opt['n_channels'] = 1
		opt['n_classes'] = 121
		opt['n_filters'] = 32
		opt['filter_gain'] = 2
		opt['augment'] = True
		opt['crop_shape'] = 10
		opt['log_path'] = './logs/deep_plankton'
		opt['checkpoint_path'] = './checkpoints/deep_plankton'
	elif opt['datasetIdx'] == 'galaxies': 
		# Load dataset
		data = load_dataset(data_dir, 'galaxies_numpy')
		opt['is_classification'] = False
		opt['dim'] = 64
		opt['n_channels'] = 3
		opt['n_classes'] = 37
	else:
		print('Dataset unrecognized, options are:')
		print('mnist, cifar10, plankton, galaxies')
		sys.exit(1)
	
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


if __name__ == '__main__':
	deviceIdxs = [int(x.strip()) for x in sys.argv[2].split(',')]
	opt = {}
	opt['model'] = sys.argv[3]
	opt['datasetIdx'] = sys.argv[1]
	opt['deviceIdxs'] = deviceIdxs
	'''
	
	opt = {}
	opt['model'] = getattr(equivariant, sys.argv[3])
	opt['lr'] = 0.00756
	opt['batch_size'] = 46
	opt['n_epochs'] = 80
	opt['n_filters'] = 8
	opt['trial_num'] = 'B'
	opt['combine_train_val'] = False
	opt['std_mult'] = 0.7
	opt['filter_gain'] = 2.1
	opt['momentum'] = 0.933
	opt['psi_preconditioner'] = 7.82
	opt['delay'] = 12
	opt['datasetIdx'] = sys.argv[1]
	opt['deviceIdxs'] = deviceIdxs
	opt['display_step'] = 10
	opt['augment'] = False
	opt['log_path'] = './logs/deep_stable2'
	opt['checkpoint_path'] = './checkpoints/deep_stable2'
	opt['save_step']
	'''

	run(opt)
	print("ALL FINISHED! :)")
