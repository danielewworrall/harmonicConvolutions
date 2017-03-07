'''SGD equivariance'''

import os
import sys
import time
sys.path.append('../')
sys.path.append('./imagenet')

import cv2
import input_data
import numpy as np
import skimage.io as skio
import tensorflow as tf

import equivariant_loss as el
import mnist_loader
import models

from spatial_transformer import transformer

################ DATA #################
def load_data():
	mnist = mnist_loader.read_data_sets("/tmp/data/", one_hot=True)
	data = {}
	data['X'] = {'train': np.reshape(mnist.train._images, (-1,28,28,1)),
					 'valid': np.reshape(mnist.validation._images, (-1,28,28,1)),
					 'test': np.reshape(mnist.test._images, (-1,28,28,1))}
	data['Y'] = {'train': mnist.train._labels,
					 'valid': mnist.validation._labels,
					 'test': mnist.test._labels}
	return data


def random_sampler(n_data, opt, random=True):
	"""Return minibatched data"""
	if random:
		indices = np.random.permutation(n_data)
	else:
		indices = np.arange(n_data)
	mb_list = []
	for i in xrange(int(float(n_data)/opt['mb_size'])):
		mb_list.append(indices[opt['mb_size']*i:opt['mb_size']*(i+1)])
	return mb_list

############## MODEL ####################
def conv(x, shape, name='0', bias_init=0.01, return_params=False):
	"""Basic convolution"""
	He_initializer = tf.contrib.layers.variance_scaling_initializer()
	W = tf.get_variable('W'+name, shape=shape, initializer=He_initializer)
	z = tf.nn.conv2d(x, W, (1,1,1,1), 'SAME', name='conv'+str(name))
	return z


def bias_add(x, nc, bias_init=0.01, name='0'):
	const_initializer = tf.constant_initializer(value=bias_init)
	b = tf.get_variable('b'+name, shape=nc, initializer=const_initializer)
	return tf.nn.bias_add(x, b)


def siamese_model(x, t_params, f_params, opt):
	"""Build siamese models for equivariance tests"""
	nc = opt['n_channels']
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('siamese') as scope:
		y1 = big_block(x, nc, opt, name='siamese')
		# Transformer branch
		y_post = el.transform_features(y1, t_params, f_params)
		
		# Siamese loss
		x_pre = transformer(x, t_params, (xsh[1],xsh[2]))
		scope.reuse_variables()
		y2 = big_block(x_pre, nc, opt, name='siamese')
		
		# Tail
	with tf.variable_scope('tail') as scope:
		y = tf.nn.max_pool(y1, (1,2,2,1), (1,2,2,1), padding='VALID')
		logits = build_tail(y, 2*nc, 4*nc, opt, name='tail')
		scope.reuse_variables()
		y_ = tf.nn.max_pool(y2, (1,2,2,1), (1,2,2,1), padding='VALID')
		t_logits = build_tail(y_, 2*nc, 4*nc, opt, name='tail')
	
	return logits, t_logits, [y_post,], [y2,]


def big_block(x, nc, opt, name='_MC'):
	# L1
	l1 = block(x, 1, nc, opt, name=name+'_1')
	l1 = tf.nn.max_pool(l1, (1,2,2,1), (1,2,2,1), padding='VALID')
	# L2
	l2 = block(l1, nc, 2*nc, opt, name=name+'_2')
	l3 = conv(l2, [1,1,2*nc,2*nc], name='3')
	return bias_add(l3, 2*nc, name=name+'_3')


def block(x, ni, nc, opt, name='block'):
	"""Build the model we want"""
	l1 = conv(x, [3,3,ni,nc], name=name+'_1')
	l1 = bias_add(l1, nc, name=name+'_1')
	l1 = tf.nn.relu(l1)
	
	l2 = conv(l1, [3,3,nc,nc], name=name+'_2')
	l2 = bias_add(l2, nc, name=name+'_2')
	return tf.nn.relu(l2)


def build_tail(x, ni, nc, opt, name='tail'):
	"""Build the model we want"""
	l1 = conv(x, [3,3,ni,nc], name='1'+name )
	l1 = bias_add(l1, nc, name='1'+name)
	l1 = tf.nn.relu(l1)
	
	l2 = conv(l1, [3,3,nc,10], name='2'+name)
	l2 = bias_add(l2, 10, name='2'+name)
	l2 = tf.reduce_mean(l2, axis=[1,2])
	return l2


######################################################


def train(inputs, outputs, ops, opt):
	"""Training loop"""
	x, labels, global_step, t_params, f_params, lr = inputs
	loss, top1, merged = outputs
	train_op = ops
	
	# For checkpoints
	saver = tf.train.Saver()
	gs = 0
	start = time.time()
	
	data = load_data()
	data['X']['train'] = data['X']['train'][:opt['train_size'],:]
	data['Y']['train'] = data['Y']['train'][:opt['train_size'],:]
	n_train = data['X']['train'].shape[0]
	n_valid = data['X']['valid'].shape[0]
	n_test = data['X']['test'].shape[0]
	with tf.Session() as sess:
		# Threading and queueing
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		# Initialize variables
		init = tf.global_variables_initializer()
		sess.run(init)
		
		train_writer = tf.summary.FileWriter(opt['summary_path'], sess.graph)
		# Training loop
		for epoch in xrange(opt['n_epochs']):
			# Learning rate
			exponent = sum([epoch > i for i in opt['lr_schedule']])
			current_lr = opt['lr']*np.power(0.1, exponent)	
			
			# Train
			train_loss = 0.
			train_acc = 0.
			# Run training steps
			mb_list = random_sampler(n_train, opt)
			for i, mb in enumerate(mb_list):
				tp, fp = el.random_transform(opt['mb_size'], opt['im_size'])
				ops = [global_step, loss, top1, merged, train_op]
				feed_dict = {x: data['X']['train'][mb,...],
								 labels: data['Y']['train'][mb,...],
								 t_params: tp,
								 f_params: fp,
								 lr: current_lr}
				gs, l, t1, summary, __ = sess.run(ops, feed_dict=feed_dict)
				train_loss += l
				train_acc += t1
				# Summary writers
				train_writer.add_summary(summary, gs)
			train_loss /= (i+1.)
			train_acc /= (i+1.)
			
			# Validation
			valid_acc = 0.
			# Run training steps
			mb_list = random_sampler(n_valid, opt, random=False)
			for i, mb in enumerate(mb_list):
				ops = top1
				feed_dict = {x: data['X']['valid'][mb,...],
								 labels: data['Y']['valid'][mb,...]}
				t1 = sess.run(ops, feed_dict=feed_dict)
				valid_acc += t1
			
			valid_acc /= (i+1)
			# Printing and checkpoint saving
			print('[{:06d} | {:06d}] Train loss: {:03f}, Train top1: {:03f}, Valid top1: {:03f}, LR: {:0.1e}' \
					.format(int(time.time()-start), epoch, train_loss, train_acc, valid_acc, current_lr))
			if epoch % opt['save_step'] == 0:
				save_path = saver.save(sess, opt['save_path'], global_step=gs)
				print("Model saved in file: %s" % save_path)
		
		# Testing
		test_acc = 0.
		# Run training steps
		mb_list = random_sampler(n_test, opt, random=False)
		for i, mb in enumerate(mb_list):
			ops = top1
			feed_dict = {x: data['X']['test'][mb,...],
							 labels: data['Y']['test'][mb,...]}
			t1 = sess.run(ops, feed_dict=feed_dict)
			test_acc += t1
		
		return test_acc / (i+1)	


def main(opt):
	"""Main loop"""
	tf.reset_default_graph()
	opt['root'] = '/home/daniel'
	dir_ = opt['root'] + '/Code/harmonicConvolutions/tensorflow1/scale'
	opt['mb_size'] = 128
	opt['n_channels'] = 10
	opt['n_epochs'] = 100
	opt['lr_schedule'] = [50, 75]
	opt['lr'] = 1e-2
	opt['save_step'] = 10
	opt['im_size'] = (28,28)
	opt['train_size'] = 2000
	#opt['equivariant_weight'] = 1e-5 #1e-3
	flag = 'bn'
	opt['summary_path'] = dir_ + '/summaries/train_{:.0e}_{:s}'.format(opt['equivariant_weight'], flag)
	opt['save_path'] = dir_ + '/checkpoints/train_{:.0e}_{:s}/model.ckpt'.format(opt['equivariant_weight'], flag)

	
	# Construct input graph
	x = tf.placeholder(tf.float32, [opt['mb_size'],28,28,1], name='x')
	labels = tf.placeholder(tf.int32, [opt['mb_size'], 10], name='labels')
	# Define variables
	global_step = tf.Variable(0, name='global_step', trainable=False)
	t_params = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
	f_params = tf.placeholder(tf.float32, [opt['mb_size'],2,2], name='f_params')
	lr = tf.placeholder(tf.float32, [], name='lr')
	# Build the model
	logits, t_logits, y, yr = siamese_model(x, t_params, f_params, opt)
	
	# Build loss and metrics
	softmax = tf.nn.softmax_cross_entropy_with_logits(logits=t_logits, labels=labels)
	classification_loss = tf.reduce_mean(softmax)
	equi_loss = 0.
	layer_equi_summaries = []
	for i, (y_, yr_) in enumerate(zip(y, yr)):
		layer_equi_loss = tf.reduce_mean(tf.square(y_ - yr_))
		equi_loss += layer_equi_loss
		layer_equi_summaries.append(tf.summary.scalar('Equivariant loss'+str(i), layer_equi_loss))
	loss = classification_loss + opt['equivariant_weight']*equi_loss
		
	# Accuracies
	logits_ = tf.argmax(logits, axis=1)
	labels_ = tf.argmax(labels, axis=1)
	top1 = tf.reduce_mean(tf.cast(tf.equal(logits_, labels_), tf.float32))
	
	loss_summary = tf.summary.scalar('Loss', loss)
	class_summary = tf.summary.scalar('Classification Loss', classification_loss)
	equi_summary = tf.summary.scalar('Equivariant loss', equi_loss)
	#reg_loss = tf.summary.scalar('Regularization loss', regularization_loss)
	top1_summary = tf.summary.scalar('Top1 Accuracy', top1)
	lr_summary = tf.summary.scalar('Learning rate', lr)
	merged = tf.summary.merge_all()
	
	# Build optimizer
	optim = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
	train_op = optim.minimize(loss, global_step=global_step)
	
	inputs = [x, labels, global_step, t_params, f_params, lr]
	outputs = [loss, top1, merged]
	ops = [train_op]
	
	# Train
	return train(inputs, outputs, ops, opt)


if __name__ == '__main__':
	opt = {}
	main(opt)
