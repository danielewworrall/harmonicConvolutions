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

def rerun_model(opt, data):
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
	#init_var = []
	#for var in tf.all_variables():
	#	if 'ExponentialMovingAverage' in var.name:
	#		init_var.append(var)
	#init = tf.initialize_variables(init_var)
	
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
	saver = tf.train.Saver()
	if opt['load_pretrained']:
		saver.restore(sess, './checkpoints/deep_bsd/trialS/model.ckpt')
	start = time.time()

def run(opt):
	opt, data = get_settings(opt)
	return rerun_model(opt, data)


if __name__ == '__main__':
	deviceIdxs = [int(x.strip()) for x in sys.argv[1].split(',')]
	opt = {}
	opt['deviceIdxs'] = deviceIdxs
	opt['data_dir'] = sys.argv[2]
	opt['machine'] = sys.argv[3]

	run(opt)
	print("ALL FINISHED! :)")
