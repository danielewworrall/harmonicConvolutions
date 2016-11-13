'''Boundary detection---suff diff to everything else to require own file'''

import os
import sys
import time

from equivariant import deep_bsd

import os
import sys
import time

import cPickle as pkl
import cv2
import equivariant
import numpy as np
import scipy as sp
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import skimage.io as skio
import tensorflow as tf

import input_data

from equivariant import *
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import misc
from steer_conv import *

###HELPER FUNCTIONS------------------------------------------------------------------
def get_loss(opt, pred, y, sl=None):
	"""Pred is a dist of feature maps and so is y"""
	cost = 0.
	beta = 1-tf.reduce_mean(y)
	pw = beta / (1. - beta)
	for key in pred.keys():
		pred_ = pred[key]
		# side-weight/fusion loss
		#mult = 1.
		#if (sl is not None) and (key == 'fuse'):
		#	mult = sl
		cost += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred_, y, pw))
	print('  Constructed loss')
	return cost

def get_io_placeholders(opt):
	"""Return placeholders for classification/regression"""
	size = int(opt['dim'])
	size2 = int(opt['dim2'])
	io_x = tf.placeholder(tf.float32, [opt['batch_size'],None,None,3])
	io_y = tf.placeholder(tf.float32, [opt['batch_size'],None,None,1], name='y')
	return io_x, io_y

def build_optimizer(cost, lr, opt):
	"""Apply the psi_precponditioner"""
	#mmtm = tf.train.MomentumOptimizer
	mmtm = tf.train.AdamOptimizer
	#optim = mmtm(learning_rate=lr, momentum=opt['momentum'], use_nesterov=True)
	optim = mmtm(learning_rate=lr)
	
	grads_and_vars = optim.compute_gradients(cost)
	modified_gvs = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			g = opt['psi_preconditioner']*g
		modified_gvs.append((g, v))
	optimizer = optim.apply_gradients(modified_gvs)
	print('  Optimizer built')
	return optimizer

def build_feed_dict(opt, io, batch, lr, pt, lr_, pt_):
	'''Build a feed_dict appropriate to training regime'''
	batch_x, batch_y, __ = batch
	fd = {lr : lr_, pt : pt_}
	bs = opt['batch_size']
	for g in xrange(len(opt['deviceIdxs'])):
		fd[io['x'][g]] = batch_x[g*bs:(g+1)*bs,:]
		fd[io['y'][g]] = batch_y[g*bs:(g+1)*bs]
	return fd

##### TRAINING LOOPS #####
def loop(mode, sess, io, opt, data, cost, lr, lr_, pt, sl=None, epoch=0,
		 optim=None, step=0, anneal=0.):
	"""Run a loop"""
	X = data[mode+'_x']
	Y = data[mode+'_y']
	is_training = (mode=='train')
	n_GPUs = len(opt['deviceIdxs'])
	generator = pklbatcher(X, Y, n_GPUs*opt['batch_size'], anneal=anneal,
						   shuffle=is_training, augment=opt['augment'],
						   img_shape=(opt['dim'], opt['dim2'], 3))
	#if is_training:
	#	generator = threadedGen(generator)
	cost_total = 0.
	for i, batch in enumerate(generator):
		fd = build_feed_dict(opt, io, batch, lr, pt, lr_, is_training)
		if sl is not None:
				fd[sl] = 1. - float(epoch)/100.
		if mode == 'train':
			__, cost_ = sess.run([optim, cost], feed_dict=fd)
		else:
			cost_ = sess.run(cost, feed_dict=fd)
		if step % opt['display_step'] == 0:
			print('  ' + mode + ' loss: %f' % cost_)
		cost_total += cost_
	return cost_total/(i+1.), step

def threadedGen(generator, num_cached=50):
    '''Threaded generator to multithread the data loading pipeline'''
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()

def construct_model_and_optimizer(opt, io, lr, pt, sl=None):
	"""Build the model and an single/multi-GPU optimizer"""
	if len(opt['deviceIdxs']) == 1:
		size = opt['dim']
		size2 = opt['dim2']
		pred, __ = opt['model'](opt, io['x'][0], pt)
		loss = get_loss(opt, pred, io['y'][0], sl=sl)
		train_op = build_optimizer(loss, lr, opt)
	return loss, train_op, pred

def save_predictions(sess, x, opt, pred, pt, data, epoch):
	"""Save predictions to output folder"""
	X = data['valid_x']
	Y = data['valid_y']
	save_path = opt['test_path'] + '/T_' + str(epoch)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	generator = pklbatcher(X, Y, opt['batch_size'], shuffle=False,
						   augment=False, img_shape=(opt['dim'], opt['dim2']))
	# Use sigmoid to map to [0,1]
	bsd_map = tf.nn.sigmoid(pred['fuse'])
	j = 0
	for batch in generator:
		batch_x, batch_y, excerpt = batch
		output = sess.run(bsd_map, feed_dict={x: batch_x, pt: False})
		for i in xrange(output.shape[0]):
			save_name = save_path + '/' + str(excerpt[i]).replace('.jpg','.png')
			im = output[i,:,:,0]
			im = (255*im).astype('uint8')
			if data['valid_x'][excerpt[i]]['transposed']:
				im = im.T
			skio.imsave(save_name, im)
			j += 1
	print('Saved predictions to: %s' % (save_path,))

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
	if opt['anneal_sl']:
		sl = tf.placeholder(tf.float32, [], name='side_loss_multiplier')
	else:
		sl = None
	
	# Construct model and optimizer
	loss, train_op, pred = construct_model_and_optimizer(opt, io, lr, pt, sl=sl)
	
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Summary writers
	tcost_ss = create_scalar_summary('training_cost')
	vcost_ss = create_scalar_summary('validation_cost')
	lr_ss = create_scalar_summary('learning_rate')
	
	# Configure tensorflow session
	config = config_init()
	if n_GPUs == 1:
		config.inter_op_parallelism_threads = 1 #prevent inter-session threads?
	sess = tf.Session(config=config)
	summary = tf.train.SummaryWriter(opt['log_path'], sess.graph)
	print('Summaries constructed...')
	
	sess.run(init)
	saver = tf.train.Saver(tf.trainable_variables())
	saver.restore(sess, './checkpoints/deep_bsd/trialY/model.ckpt')
	start = time.time()
	lr_ = opt['lr']
	epoch = 0
	step = 0.
	counter = 0
	best = -1e6
	bs = opt['batch_size']
	print('Starting training loop...')
	while epoch < opt['n_epochs']:
		# Need batch_size*n_GPUs amount of data
		anneal = 0.1 + np.minimum(epoch/30.,1.)
		cost_total, step = loop('train', sess, io, opt, data, loss, lr, lr_, pt,
								sl=sl, epoch=epoch, optim=train_op, step=step,
								anneal=anneal)
		
		vloss_total, __ = loop('valid', sess, io, opt, data, loss, lr, lr_, pt,
							   sl=sl, epoch=epoch, optim=train_op, anneal=1.)
		
		fd = {tcost_ss[0] : cost_total, vcost_ss[0] : vloss_total,
			  lr_ss[0] : lr_}
		summaries = sess.run([tcost_ss[1], vcost_ss[1], lr_ss[1]], feed_dict=fd)
		for summ in summaries:
			summary.add_summary(summ, step)
		best, counter, lr_ = get_learning_rate(opt, -vloss_total, best, counter, lr_)
		#lr_ = opt['lr'] / (1. + np.sqrt(epoch/2.))
		
		print "[" + str(opt['trial_num']),str(epoch) + \
		"] Time: " + \
		"{:.3f}".format(time.time()-start) + ", Counter: " + \
		"{:d}".format(counter) + ", Loss: " + \
		"{:.5f}".format(cost_total) + ", Val loss: " + \
		"{:.5f}".format(vloss_total)
		
		# Write test time predictions to file
		if epoch % opt['save_test_step'] == 0:
			save_predictions(sess, io['x'][0], opt, pred, pt, data, epoch)
	
		epoch += 1
	
		if (epoch) % opt['save_step'] == 0:
			save_path = saver.save(sess, opt['checkpoint_path'])
			print("Model saved in file: %s" % save_path)
			
	# Save model and exit
	save_path = saver.save(sess, opt['checkpoint_path'])
	print("Model saved in file: %s" % save_path)
	sess.close()
	return 

def load_pkl(dir_name, subdir_name, prepend=''):
	"""Load dataset from subdirectory"""
	data_dir = dir_name + '/' + subdir_name
	data = {}
	with open(data_dir + '/' + prepend + 'train_images.pkl') as fp:
		data['train_x'] = pkl.load(fp)
	with open(data_dir + '/' + prepend + 'train_labels.pkl') as fp:
		data['train_y'] = pkl.load(fp)
	with open(data_dir + '/' + prepend + 'valid_images.pkl') as fp:
		data['valid_x'] = pkl.load(fp)
	with open(data_dir + '/' + prepend + 'valid_labels.pkl') as fp:
		data['valid_y'] = pkl.load(fp)
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
	tf.reset_default_graph()
	
	# Default configuration
	opt['trial_num'] = 'E'
	opt['combine_train_val'] = False	
	
	data = load_pkl(opt['data_dir'], 'bsd_pkl_float', prepend='')
	opt['model'] = getattr(equivariant, 'deep_bsd')
	opt['is_bsd'] = True
	opt['lr'] = 1e-3
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
	opt['n_filters'] = 16 #32
	opt['filter_gain'] = 2
	opt['augment'] = True
	opt['lr_div'] = 10.
	opt['log_path'] = './logs/deep_bsd'
	opt['checkpoint_path'] = './checkpoints/deep_bsd'
	opt['test_path'] = './bsd/trial' + opt['trial_num']
	opt['anneal_sl'] = False
	if not os.path.exists(opt['test_path']):
		os.mkdir(opt['test_path'])
	opt['save_test_step'] = 5
	
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
	deviceIdxs = [int(x.strip()) for x in sys.argv[1].split(',')]
	opt = {}
	opt['deviceIdxs'] = deviceIdxs
	opt['data_dir'] = sys.argv[2]
	opt['machine'] = sys.argv[3]

	run(opt)
	print("ALL FINISHED! :)")
