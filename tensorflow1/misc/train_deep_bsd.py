'''Boundary detection---suff diff to everything else to require own file'''

import os
import sys
import time

import cPickle as pkl
import numpy as np
import scipy as sp
import scipy.ndimage.interpolation as sciint
import skimage.io as skio
from scipy import ndimage
from scipy import misc

import tensorflow as tf

###HELPER FUNCTIONS------------------------------------------------------------------
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
	loss, train_op, pred = bsd_construct_model_and_optimizer(opt, io, lr, pt, sl=sl)
	
	# Initializing the variables
	init = tf.global_variables_initializer()
	
	# Summary writers
	tcost_ss = create_scalar_summary('training_cost')
	vcost_ss = create_scalar_summary('validation_cost')
	lr_ss = create_scalar_summary('learning_rate')
	
	# Configure tensorflow session
	config = config_init()
	if n_GPUs == 1:
		config.inter_op_parallelism_threads = 1 #prevent inter-session threads?
	sess = tf.Session(config=config)
	summary = tf.summary.FileWriter(opt['log_path'], sess.graph)
	print('Summaries constructed...')
	
	sess.run(init)
	saver = tf.train.Saver()
	if opt['load_pretrained']:
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
		cost_total, step = bsd_loop('train', sess, io, opt, data, loss, lr, lr_, pt,
								sl=sl, epoch=epoch, optim=train_op, step=step,
								anneal=anneal)
		
		vloss_total, __ = bsd_loop('valid', sess, io, opt, data, loss, lr, lr_, pt,
							   sl=sl, epoch=epoch, optim=train_op, anneal=1.)
		
		fd = {tcost_ss[0] : cost_total, vcost_ss[0] : vloss_total,
			  lr_ss[0] : lr_}
		summaries = sess.run([tcost_ss[1], vcost_ss[1], lr_ss[1]], feed_dict=fd)
		for summ in summaries:
			summary.add_summary(summ, step)
		#best, counter, lr_ = get_learning_rate(opt, -vloss_total, best, counter, lr_)
		if epoch % 40 == 39:
			lr_ = lr_ / 10.
		#lr_ = opt['lr'] / (1. + np.sqrt(epoch/2.))
		
		print "[" + str(opt['trial_num']),str(epoch) + \
		"] Time: " + \
		"{:.3f}".format(time.time()-start) + ", Counter: " + \
		"{:d}".format(counter) + ", Loss: " + \
		"{:.5f}".format(cost_total) + ", Val loss: " + \
		"{:.5f}".format(vloss_total)
		
		# Write test time predictions to file
		if epoch % opt['save_test_step'] == 0:
			bsd_save_predictions(sess, io['x'][0], opt, pred, pt, data, epoch)
	
		epoch += 1
	
		if (epoch) % opt['save_step'] == 0:
			save_path = saver.save(sess, opt['checkpoint_path'])
			print("Model saved in file: %s" % save_path)
			
	# Save model and exit
	save_path = saver.save(sess, opt['checkpoint_path'])
	print("Model saved in file: %s" % save_path)
	sess.close()
	return 

def bsd_run(opt):
	opt, data = get_settings(opt)
	return train_model(opt, data)


if __name__ == '__main__':
	deviceIdxs = [int(x.strip()) for x in sys.argv[1].split(',')]
	opt = {}
	opt['deviceIdxs'] = deviceIdxs
	opt['data_dir'] = sys.argv[2]
	opt['machine'] = sys.argv[3]

	bsd_run(opt)
	print("ALL FINISHED! :)")
