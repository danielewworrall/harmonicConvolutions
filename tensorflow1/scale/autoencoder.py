'''Autoencoder'''

import os
import sys
import time
sys.path.append('../')

#import cv2
#import input_data
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
def linear(x, shape, name='0', bias_init=0.01):
	"""Basic linea matmul layer"""
	He_initializer = tf.contrib.layers.variance_scaling_initializer()
	W = tf.get_variable(name+'_W', shape=shape, initializer=He_initializer)
	z = tf.matmul(x, W, name='mul'+str(name))
	return bias_add(z, shape[1], name=name)


def bias_add(x, nc, bias_init=0.01, name='0'):
	const_initializer = tf.constant_initializer(value=bias_init)
	b = tf.get_variable(name+'_b', shape=nc, initializer=const_initializer)
	return tf.nn.bias_add(x, b)


def siamese_model(x, t_params, f_params, opt):
	"""Build siamese models for equivariance tests"""
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('siamese') as scope:
		# Basic branch
		x1 = tf.reshape(x, tf.stack([xsh[0],784]))
		z1, r1 = autoencoder(x1)
		z1 = el.feature_space_transform2d(z1, [xsh[0], 250], f_params)
		r1 = tf.reshape(r1, tf.stack([xsh[0],28,28,1]))
		
		# Transformer branch
		scope.reuse_variables()
		x2 = transformer(x, t_params, (28,28))
		x2_ = tf.reshape(x2, tf.stack([xsh[0],784]))
		z2, r2 = autoencoder(x2_)
		r2 = tf.reshape(r2, tf.stack([xsh[0],28,28,1]))

	return x2, r1, r2, z1, z2


def single_model(x, f_params):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('siamese', reuse=True) as scope:
		# Basic branch
		x = tf.reshape(x, tf.stack([xsh[0],784]))
		with tf.variable_scope("Encoder", reuse=True) as scope:
			z = encoder(x)
		z = el.feature_space_transform2d(z, [xsh[0], 250], f_params)
		with tf.variable_scope("Decoder", reuse=True) as scope:
			r = decoder(z)
	return r


def autoencoder(x):
	"""Build autoencoder"""
	xsh = x.get_shape().as_list()
	with tf.variable_scope("Encoder") as scope:
		z = encoder(x)
	with tf.variable_scope("Decoder") as scope:
		r = decoder(z)
	return z, r


def encoder(x):
	"""Encoder MLP"""
	l1 = linear(x, [784,500], name='1')
	l2 = linear(tf.nn.relu(l1), [500,500], name='2')
	l3 = linear(tf.nn.relu(l2), [500,250], name='3')
	return linear(tf.nn.relu(l3), [250,250], name='4')

def decoder(z):
	"""Encoder MLP"""
	l1 = linear(z, [250,250], name='5')
	l2 = linear(tf.nn.relu(l1), [250,500], name='6')
	l3 = linear(tf.nn.relu(l2), [500,500], name='7')
	return tf.nn.sigmoid(linear(tf.nn.relu(l3), [500,784], name='8'))
	

######################################################


def train(inputs, outputs, ops, opt):
	"""Training loop"""
	x, global_step, t_params, f_params, lr, xs, fs_params = inputs
	loss, merged, recon = outputs
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
				ops = [global_step, loss, merged, train_op]
				feed_dict = {x: data['X']['train'][mb,...],
								 t_params: tp,
								 f_params: fp,
								 lr: current_lr}
				gs, l, summary, __ = sess.run(ops, feed_dict=feed_dict)
				train_loss += l
				# Summary writers
				train_writer.add_summary(summary, gs)
			train_loss /= (i+1.)
			print('[{:03d}]: {:03f}'.format(epoch, train_loss))
			
			# Validation
			if epoch % 10 == 0:
				Recon = []
				sample = data['X']['valid'][np.newaxis,np.random.randint(5000),...]
				max_angles = 20*20
				for i in xrange(max_angles):
					fp = el.get_f_transform(2.*np.pi*i/(1.*max_angles))[np.newaxis,:,:]
					#print fp
					ops = recon
					feed_dict = {xs: sample, fs_params: fp}
					Recon.append(sess.run(ops, feed_dict=feed_dict))
				
				samples_ = np.reshape(Recon, (-1,28,28))
				
				ns = 20
				sh = 28
				tile_image = np.zeros((ns*sh,ns*sh))
				for j in xrange(ns*ns):
					m = sh*(j/ns) 
					n = sh*(j%ns)
					tile_image[m:m+sh,n:n+sh] = 1.-samples_[j,...]
				save_name = './samples/image_%04d.png' % epoch
				skio.imsave(save_name, tile_image)


def main(opt):
	"""Main loop"""
	tf.reset_default_graph()
	opt['root'] = '/home/sgarbin'
	dir_ = opt['root'] + '/Projects/harmonicConvolutions/tensorflow1/scale'
	opt['mb_size'] = 128
	opt['n_channels'] = 10
	opt['n_epochs'] = 10000
	opt['lr_schedule'] = [50, 75]
	opt['lr'] = 1e-3
	opt['save_step'] = 10
	opt['im_size'] = (28,28)
	opt['train_size'] = 55000
	opt['equivariant_weight'] = 1e0 #1e-3
	flag = 'bn'
	opt['summary_path'] = dir_ + '/summaries/autotrain_{:.0e}_{:s}'.format(opt['equivariant_weight'], flag)
	opt['save_path'] = dir_ + '/checkpoints/autotrain_{:.0e}_{:s}/model.ckpt'.format(opt['equivariant_weight'], flag)

	
	# Construct input graph
	x = tf.placeholder(tf.float32, [opt['mb_size'],28,28,1], name='x')
	xs = tf.placeholder(tf.float32, [1,28,28,1], name='xs')
	# Define variables
	global_step = tf.Variable(0, name='global_step', trainable=False)
	t_params = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
	f_params = tf.placeholder(tf.float32, [opt['mb_size'],2,2], name='f_params')
	lr = tf.placeholder(tf.float32, [], name='lr')
	fs_params = tf.placeholder(tf.float32, [1,2,2], name='fs_params')
	# Build the model
	x_, r, r_, zt, z_ = siamese_model(x, t_params, f_params, opt)
	recon = single_model(xs, fs_params)
	
	# Build loss and metrics
	branch1_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - r), axis=(1,2)))
	branch2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_ - r_), axis=(1,2)))
	equi_loss = tf.reduce_mean(tf.reduce_sum(tf.square(zt - z_), axis=1))
	loss = branch1_loss + branch2_loss + opt['equivariant_weight']*equi_loss
	
	loss_summary = tf.summary.scalar('Loss', loss)
	equi_summary = tf.summary.scalar('Equivariant loss', equi_loss)
	lr_summary = tf.summary.scalar('Learning rate', lr)
	merged = tf.summary.merge_all()
	
	# Build optimizer
	#optim = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
	optim = tf.train.AdamOptimizer(lr)
	train_op = optim.minimize(loss, global_step=global_step)
	
	inputs = [x, global_step, t_params, f_params, lr, xs, fs_params]
	outputs = [loss, merged, recon]
	ops = [train_op]
	
	# Train
	return train(inputs, outputs, ops, opt)


if __name__ == '__main__':
	opt = {}
	main(opt)
