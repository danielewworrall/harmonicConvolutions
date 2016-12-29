import os
import time

import numpy as np
import scipy as sp

import tensorflow as tf

from io_helpers import *
from harmonic_network_models import *

#----------HELPER FUNCTIONS----------
def print_train_validation(trial_num, counter, epoch, time,
	cost_total, validation_loss_total, acc_total, validation_acc_total):
	"""Formats print-out for the training-loop
	
	"""
	print "[" + str(trial_num),str(epoch) + \
		"] Time: " + \
		"{:.3f}".format(time) + ", Counter: " + \
		"{:d}".format(counter) + ", Loss: " + \
		"{:.5f}".format(cost_total) + ", Val loss: " + \
		"{:.5f}".format(validation_loss_total) + ", Train Acc: " + \
		"{:.5f}".format(acc_total) + ", Val acc: " + \
		"{:.5f}".format(validation_acc_total)


def print_validation(trial_num, counter, epoch, time,
	cost_total, acc_total,):
	"""Formats print-out for the training-loop

	"""
	print "[" + str(trial_num),str(epoch) + \
		"] Time: " + \
		"{:.3f}".format(time) + ", Counter: " + \
		"{:d}".format(counter) + ", Loss: " + \
		"{:.5f}".format(cost_total) + ", Train Acc: " + \
		"{:.5f}".format(acc_total)


def average_gradients(gpu_grads):
	"""Calculate the average gradient for each shared variable across all gpus.
	This forces synchronisation as on the CPU if the original variables are
	defined in host memory (and needs a host2devicecopy and back).

	gpu_grads: List of lists of (gradient, variable) tuples. The outer list
	is over individual gradients. The inner list is over the gradient
	calculation for each tower.
	
	Returns:
	List of pairs of (gradient, variable) where the gradient has been averaged
	across all gpus.
	"""
	if len(gpu_grads) == 1:
		return gpu_grads[0]
	else:
		print('Processing %d sets of gradients.' % len(gpu_grads))
	average_grads = []
	for grad_and_vars in zip(*gpu_grads): #for each grad, vars set
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
		# Average over the 'gpu' dimension.
		grad = tf.concat(0, grads)
		grad = tf.reduce_mean(grad, 0)
		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads


def sparsity_regularizer(x, sparsity):
	"""Define a sparsity regularizer"""
	q = tf.reduce_mean(tf.nn.sigmoid(x))
	return -sparsity*tf.log(q) - (1-sparsity)*tf.log(1-q)


def get_loss(opt, pred, y):
	"""Constructs loss different for regression/classification

	opt: options
	pred: predictions
	y: target values

	Returns:
	Tensorflow node for calculating the final cost"""
	if opt['is_bsd']:
		cost = 0.
		beta = 1-tf.reduce_mean(y)
		pw = beta / (1. - beta)
		sparsity_coefficient = opt['sparsity']
		for key in pred.keys():
			pred_ = pred[key]
			cost += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred_, y, pw))
			# Sparsity regularizer
			cost += sparsity_coefficient*sparsity_regularizer(pred_, 1-beta)
	else:
		if opt['is_classification']:
			cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
		else:
			cost = 0.5*tf.reduce_mean(tf.pow(y - pred, 2))
	
	print('Constructed loss...')
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
	if opt['is_bsd']:
		io_x = tf.placeholder(tf.float32, [opt['batch_size'],None,None,3])
		io_y = tf.placeholder(tf.float32, [opt['batch_size'],None,None,1], name='y')
	return io_x, io_y

def build_optimizer(cost, lr, opt):
	"""Apply the psi_preconditioner"""
	optim = tf.train.AdamOptimizer(learning_rate=lr)
	grads_and_vars = optim.compute_gradients(cost)
	modified_gvs = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			g = opt['psi_preconditioner']*g
		modified_gvs.append((g, v))
	optimizer = optim.apply_gradients(modified_gvs)
	print('  Optimizer built...')
	return optimizer

def get_evaluation(pred, y, opt):
	if opt['is_bsd']:
		correct_pred = tf.equal(pred, y)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	else:		
		if opt['is_classification']:
			correct_pred = tf.equal(tf.argmax(pred, 1), y)
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		else:
			accuracy = cost
	return accuracy

def build_feed_dict(opt, batch, tf_nodes, is_training):
	'''Build a feed_dict appropriate to training regime'''
	batch_x, batch_y = batch
	fd = {tf_nodes['learning_rate'] : opt['lr'], tf_nodes['train_phase'] : is_training}
	bs = opt['batch_size']
	for g in xrange(len(opt['deviceIdxs'])):
		fd[tf_nodes['io']['x'][g]] = batch_x[g*bs:(g+1)*bs,:]
		fd[tf_nodes['io']['y'][g]] = batch_y[g*bs:(g+1)*bs]
	return fd


def loop(mode, sess, opt, data, tf_nodes, step=0):
	"""Runs the training loop
	
	mode: 'train' / 'valid' / 'test'
	sess: tf-session
	opt: opts dictionary
	data: data dict
	tf_nodes: dict of tensorflow constructed with build_model()
	step: optional parameter specifying global step

	Returns: cost, accuracy, new step 	for current epoch
	"""
	X = data[mode+'_x']
	Y = data[mode+'_y']
	is_training = (mode=='train')
	n_GPUs = len(opt['deviceIdxs'])
	generator = minibatcher(X, Y, n_GPUs*opt['batch_size'], shuffle=is_training,
									augment=opt['augment'],
									img_shape=(opt['dim'],opt['dim'],opt['n_channels']),
									crop_shape=opt['aug_crop'])
	cost_total = 0.
	acc_total = 0.
	for i, batch in enumerate(generator):
		fd = build_feed_dict(opt, batch, tf_nodes, is_training)
		if mode == 'train':
			__, cost_, acc_ = sess.run([tf_nodes['train_op'], tf_nodes['loss'], tf_nodes['accuracy']], feed_dict=fd)
		else:
			cost_, acc_ = sess.run([tf_nodes['loss'], tf_nodes['accuracy']], feed_dict=fd)
		if step % opt['display_step'] == 0:
			sys.stdout.write('  ' + mode + ' Acc.: %f\r' % acc_)
			sys.stdout.flush()
		cost_total += cost_
		acc_total += acc_
		step += 1
	return cost_total/(i+1.), acc_total/(i+1.), step

def bsd_loop(mode, sess, opt, data, tf_nodes, step=0,
		sl=None, epoch=0, anneal=0.):
	"""Run a loop"""
	X = data[mode+'_x']
	Y = data[mode+'_y']
	is_training = (mode=='train')
	n_GPUs = len(opt['deviceIdxs'])
	generator = pklbatcher(X, Y, n_GPUs*opt['batch_size'], anneal=anneal,
						   shuffle=is_training, augment=opt['augment'],
						   img_shape=(opt['dim'], opt['dim2'], 3))
	cost_total = 0.
	for i, batch in enumerate(generator):
		fd = build_feed_dict(opt, batch, tf_nodes, is_training)
		if sl is not None:
				fd[sl] = np.maximum(1. - float(epoch)/100.,0.)
		if mode == 'train':
			__, cost_ = sess.run([tf_nodes['train_op'], tf_nodes['loss']], feed_dict=fd)
		else:
			cost_ = sess.run(tf_nodes['loss'], feed_dict=fd)
		if step % opt['display_step'] == 0:
			print('  ' + mode + ' loss: %f' % cost_)
		cost_total += cost_
	return cost_total/(i+1.), step

def construct_model_and_optimizer(opt, tf_nodes):
	"""Build the model and an single/multi-GPU optimizer
	
	opt: options dict
	tf_nodes: dict of tf nodes constructed with build_model()

	Returns:
	cost, accuracy, training_op
	"""
	if len(opt['deviceIdxs']) == 1:
		pred = opt['model'](opt, tf_nodes['io']['x'][0], tf_nodes['train_phase'])
		loss = get_loss(opt, pred, tf_nodes['io']['y'][0])
		accuracy = get_evaluation(pred, tf_nodes['io']['y'][0], opt)
		train_op = build_optimizer(loss, tf_nodes['learning_rate'], opt)
	else:
		# Multi_GPU Optimizer
		optim = tf.train.MomentumOptimizer(learning_rate=tf_nodes['learning_rate'],
			momentum=opt['momentum'], use_nesterov=True)
		#setup model for each GPU
		linearGPUIdx = 0
		gradientsPerGPU = []
		lossesPerGPU = []
		accuracyPerGPU = []
		for g in opt['deviceIdxs']: #for every specified device
			with tf.device('/gpu:%d' % g): #create a copy of the network
				print('Building Model on GPU: %d' % g)
				with tf.name_scope('%s_%d' % (opt['model'].__name__, 0)) as scope:
					# Forward pass
					pred = opt['model'](opt, tf_nodes['io']['x'][linearGPUIdx], tf_nodes['train_phase'])
					loss = get_loss(opt, pred, tf_nodes['io']['y'][linearGPUIdx])
					accuracy = get_evaluation(pred, tf_nodes['io']['y'][linearGPUIdx], opt)
					# Reuse variables for the next tower
					tf.get_variable_scope().reuse_variables()
					# Calculate gradients for minibatch on this gpus
					grads = optim.compute_gradients(loss)
					# Keep track of gradients/losses/accuracies across all gpus
					gradientsPerGPU.append(grads)
					lossesPerGPU.append(loss)
					accuracyPerGPU.append(accuracy)
			linearGPUIdx += 1
		# CPU-side synchronisation 
		# Invoking CudaDevice2Host copy and averaging host-side forces synchronisation
		# across all devices
		grads = average_gradients(gradientsPerGPU)

		apply_gradient_op = optim.apply_gradients(grads)
		train_op = tf.group(apply_gradient_op)
		loss = tf.reduce_mean(tf.pack(lossesPerGPU, axis=0))
		accuracy = tf.reduce_mean(tf.pack(accuracyPerGPU, axis=0))
	return loss, accuracy, train_op


def build_model(opt, data):
	"""Builds model and optimiser nodes
	
	opt: dict of options
	data: dict of numpy data

	Returns a dict containing: 'learning_rate', 'train_phase', 'loss'
	'accuracy', 'train_op', and IO placeholders 'x', 'y'
	"""
	n_GPUs = len(opt['deviceIdxs'])
	print('Using Multi-GPU Model with %d devices.' % n_GPUs)
	#tensorflow nodes
	tf_nodes = {}
	tf_nodes['io'] = {}
	tf_nodes['io']['x'] = []
	tf_nodes['io']['y'] = []
	for g in opt['deviceIdxs']:
		with tf.device('/gpu:%d' % g):
			io_x, io_y = get_io_placeholders(opt)
			tf_nodes['io']['x'].append(io_x)
			tf_nodes['io']['y'].append(io_y)
	tf_nodes['learning_rate'] = tf.placeholder(tf.float32, name='learning_rate')
	tf_nodes['train_phase'] = tf.placeholder(tf.bool, name='train_phase')
	
	# Construct model and optimizer
	tf_nodes['loss'], tf_nodes['accuracy'], tf_nodes['train_op'] = construct_model_and_optimizer(opt, tf_nodes)

	tf_nodes['sum'] = {}
	tf_nodes['sum']['train_cost'] = create_scalar_summary('training_cost')
	tf_nodes['sum']['val_cost'] = create_scalar_summary('validation_cost')
	tf_nodes['sum']['val_acc'] = create_scalar_summary('validation_accuracy')
	tf_nodes['sum']['learning_rate'] = create_scalar_summary('learning_rate')
	return tf_nodes


def train_model(opt, data, tf_nodes):
	"""Generalized training function
	
	opt: dict of options
	data: dict of numpy data
	tf_nodes: dict of nodes initialised in build_model()
	"""
	n_GPUs = len(opt['deviceIdxs'])
	print('Using Multi-GPU Model with %d devices.' % n_GPUs)

	# Initializing the variables
	init = tf.initialize_all_variables()
	if opt['combine_train_val']:
		data['train_x'] = np.vstack([data['train_x'], data['valid_x']])
		data['train_y'] = np.hstack([data['train_y'], data['valid_y']])

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
	epoch = 0
	step = 0.
	counter = 0
	best = 0.
	print('Starting training loop...')
	while epoch < opt['n_epochs']:
		# Need batch_size*n_GPUs amount of data
		cost_total, acc_total, step = loop('train', sess, opt, data, tf_nodes,step=step)
		if not opt['combine_train_val']:
			vloss_total, vacc_total, __ = loop('valid', sess, opt, data, tf_nodes)
			#build the feed-dict
			fd = {tf_nodes['sum']['train_cost'][0] : cost_total,
				tf_nodes['sum']['val_cost'][0] : vloss_total,
				tf_nodes['sum']['val_acc'] [0] : vacc_total,
				tf_nodes['sum']['learning_rate'][0] : opt['lr']}
			summaries = sess.run([tf_nodes['sum']['train_cost'][1], tf_nodes['sum']['val_cost'][1],
					tf_nodes['sum']['val_acc'] [1], tf_nodes['sum']['learning_rate'][1]],
				feed_dict=fd)
			for summ in summaries:
				summary.add_summary(summ, step)
			best, counter, opt['lr'] = get_learning_rate(opt, vacc_total, best, counter, opt['lr'])
			print_train_validation(opt['trial_num'], counter, epoch, time.time()-start,
				cost_total, vloss_total, acc_total, vacc_total)
		else:
			best, counter, opt['lr'] = get_learning_rate(opt, acc_total, best, counter, opt['lr'])
			print_validation(opt['trial_num'], counter, epoch, time.time()-start, cost_total, acc_total)
		epoch += 1

		if (epoch) % opt['save_step'] == 0:
			save_path = saver.save(sess, opt['checkpoint_path'])
			print("Model saved in file: %s" % save_path)
			
	if (opt['datasetIdx'] == 'plankton') or (opt['datasetIdx'] == 'galaxies'):
		tacc_total = vacc_total
	else:
		print('Testing')
		__, tacc_total, __ = loop('test', sess, opt, data, tf_nodes)
		print('Test accuracy: %f' % (tacc_total,))
	
	# Save model and exit
	save_path = saver.save(sess, opt['checkpoint_path'])
	print("Model saved in file: %s" % save_path)
	sess.close()
	return tacc_total


def create_scalar_summary(name):
	"""Create a scalar summary placeholder and op"""
	ss = []
	ss.append(tf.placeholder(tf.float32, [], name=name))
	ss.append(tf.scalar_summary(name+'_summary', ss[0]))
	return ss

def config_init():
	"""Default config settings. Prevents excessive memory usage
	This is not neccessarily optimal for memory fragmentation,
	but we found it more convenient on a multi-GPU system with more than
	one user (and no changing memory requirements).
	"""
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = False
	return config

#----------Main Entry Point for Training----------
def build_all_and_train(opt, data):
	# Check that save paths exist
	opt['log_path'] = opt['log_path'] + '/trial' + str(opt['trial_num'])
	opt['checkpoint_path'] = opt['checkpoint_path'] + '/trial' + \
							str(opt['trial_num']) 
	if not os.path.exists(opt['log_path']):
		print('Creating log path')
		os.makedirs(opt['log_path'])
	if not os.path.exists(opt['checkpoint_path']):
		print('Creating checkpoint path')
		os.makedirs(opt['checkpoint_path'])
	opt['checkpoint_path'] = opt['checkpoint_path'] + '/model.ckpt'
	
	# Print out options
	print('Specified Options:')
	for key, val in opt.iteritems():
		print(key + ': ' + str(val))
	# Parameters
	tf.reset_default_graph()
	#build the model
	tf_nodes = build_model(opt, data)
	print('Successfully built model...')
	#train it
	return train_model(opt, data, tf_nodes)
