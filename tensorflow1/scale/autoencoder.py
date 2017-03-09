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
'''
#-----------ARGS----------
flags = tf.app.flags
FLAGS = flags.FLAGS
#execution modes
flags.DEFINE_boolean('train', False, 'trains the model')
#IO
flags.DEFINE_boolean('SAVE', True, 'saves the model')
flags.DEFINE_boolean('RESTORE', False, 'restores a model from disk')
flags.DEFINE_integer('save_interval', 1000, '')

#data params
flags.DEFINE_integer('width', 96, '')
flags.DEFINE_integer('height', 96, '')
#training params
flags.DEFINE_float('learning_rate', 1e-3, '')
flags.DEFINE_integer('num_iterations', 10000000, '')
flags.DEFINE_integer('batch_size', 128, '')
##---------------------
'''

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
def dot_product(x, y):
	return tf.reduce_sum(tf.multiply(x,y), axis=1)

def cosine_distance(x, y):
	return tf.reduce_mean(dot_product(x, y) / (dot_product(x, x) * dot_product(y, y)))

def convolutional(x, conv_size, num_filters, stride=1, name='c0',
		bias_init=0.01, padding='SAME', non_linear_func=tf.nn.relu):
	w = tf.get_variable(name + '_conv_w', [conv_size, conv_size, x.get_shape()[3], num_filters])
	b = tf.get_variable(name + '_conv_b', [num_filters])
	result = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding, name = name + 'conv')
	return non_linear_func(bias_add(result, num_filters, name=name), name = name + '_conv_nl')

def deconvolutional_deepmind(x, conv_size, out_shape, stride=1, name='c0',
		bias_init=0.01, padding='SAME', non_linear_func=tf.nn.relu):
	#resize images
	result = tf.image.resize_images(x, [out_shape[1], out_shape[2]],
		method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	#perform convolution
	result = convolutional(result, conv_size, out_shape[3], stride=stride,
		name=name, bias_init=bias_init, padding=padding, non_linear_func=non_linear_func)
	return result


def flatten(input):
	s = input.get_shape().as_list()
	num_params = 1
	for i in range(1, len(s)): #ignore batch size
		num_params *= s[i]
	return tf.reshape(input, [s[0], num_params]), num_params

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


def single_model(x, f_params, name_scope='siamese', conv=False, t_params=[]):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope(name_scope, reuse=True) as scope:
		# Basic branch
		x = tf.reshape(x, tf.stack([xsh[0],784]))
		with tf.variable_scope("Encoder", reuse=True) as scope:
			z = encoder(x, conv=conv)
		if conv:
			z = el.transform_features(z, t_params, f_params)
		else:
			z = el.feature_space_transform2d(z, [xsh[0], 128], f_params)
		with tf.variable_scope("Decoder", reuse=True) as scope:
			r = decoder(z, conv=conv)
	return r

def single_model_non_siamese(x, f_params, conv=False, t_params=[]):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('mainModel', reuse=False) as scope:
		# Basic branch
		x = tf.reshape(x, tf.stack([xsh[0],784]))
		with tf.variable_scope("Encoder", reuse=False) as scope:
			z = encoder(x, conv=conv)
		if conv:
			z = el.transform_features(z, t_params, f_params)
		else:
			z = el.feature_space_transform2d(z, [xsh[0], 128], f_params)
		with tf.variable_scope("Decoder", reuse=False) as scope:
			r = decoder(z, conv=conv)
	return r, z


def autoencoder(x):
	"""Build autoencoder"""
	xsh = x.get_shape().as_list()
	with tf.variable_scope("Encoder") as scope:
		z = encoder(x, conv=False)
	with tf.variable_scope("Decoder") as scope:
		r = decoder(z)
	return z, r


def encoder(x, conv=False):
	if conv:
		print('encoder activation sizes:')
		stream = tf.reshape(x, [x.get_shape().as_list()[0], 28, 28, 1])
		print(stream.get_shape().as_list())
		stream = convolutional(stream, 3, 16, stride=2, name='c1')
		print(stream.get_shape().as_list())
		stream = convolutional(stream, 3, 32, stride=2, name='c2')
		print(stream.get_shape().as_list())
		stream = convolutional(stream, 3, 64, stride=2, name='c3')
		print(stream.get_shape().as_list())
		return stream
	else:
		"""Encoder MLP"""
		l1 = linear(x, [784,512], name='1')
		#l2 = linear(tf.nn.relu(l1), [500,500], name='2')
		#l2 = linear(tf.nn.relu(l2), [500,500], name='2b')
		#l2 = linear(tf.nn.relu(l2), [500,500], name='2c')
		#l3 = linear(tf.nn.relu(l2), [500,256], name='3')
		return linear(tf.nn.sigmoid(l1), [512,128], name='4')

def decoder(z, conv=False):
	bs = z.get_shape()[0]
	if conv:
		#stream = deconvolutional_deepmind(z, 3, [bs, 4, 4, 32], name='c13')
		stream = deconvolutional_deepmind(z, 3, [bs, 7, 7, 32], name='c13')
		stream = deconvolutional_deepmind(stream, 3, [bs, 14, 14, 16], name='c14')
		stream = deconvolutional_deepmind(stream, 3, [bs, 28, 28, 1], name='c15')
		return stream
	else:
		"""Encoder MLP"""
		l1 = linear(z, [128,512], name='5')
		#l2 = linear(tf.nn.relu(l1), [256,500], name='6')
		#l3 = linear(tf.nn.relu(l2), [500,500], name='7')
		#l3 = linear(tf.nn.relu(l3), [500,500], name='7b')
		#l3 = linear(tf.nn.relu(l3), [500,500], name='7c')
		return tf.nn.sigmoid(linear(tf.nn.sigmoid(l1), [512,784], name='8'))
	

######################################################

def train(inputs, outputs, ops, opt):
	"""Training loop"""
	x, global_step, t_params_initial, t_params, f_params, lr, xs, fs_params = inputs
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
				#initial random transform
				tp_init, fp_init = el.random_transform(opt['mb_size'], opt['im_size'])

				tp, fp = el.random_transform(opt['mb_size'], opt['im_size'])
				ops = [global_step, loss, merged, train_op]
				feed_dict = {x: data['X']['train'][mb,...],
								 t_params: tp,
								 f_params: fp,
								 t_params_initial: tp_init,
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
				#pick a random initial transformation
				tp_init, fp_init = el.random_transform(opt['mb_size'], opt['im_size'])
				for i in xrange(max_angles):
					fp = el.get_f_transform(2.*np.pi*i/(1.*max_angles))[np.newaxis,:,:]
					#print fp
					ops = recon
					feed_dict = {xs: sample,
								fs_params: fp,
								t_params: tp_init}
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
	opt['equivariant_weight'] = 1 #1e-3
	flag = 'bn'
	opt['summary_path'] = dir_ + '/summaries/autotrain_{:.0e}_{:s}'.format(opt['equivariant_weight'], flag)
	opt['save_path'] = dir_ + '/checkpoints/autotrain_{:.0e}_{:s}/model.ckpt'.format(opt['equivariant_weight'], flag)
	opt['loss_type_image'] = 'l2'
	opt['convolutional'] = True

	# Construct input graph
	#input image
	x = tf.placeholder(tf.float32, [opt['mb_size'],28,28,1], name='x')
	#input for validation
	xs = tf.placeholder(tf.float32, [1,28,28,1], name='xs')
	# Define variables
	global_step = tf.Variable(0, name='global_step', trainable=False)
	#initial transformation
	t_params_initial = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
	#transform corresponding to latents
	t_params = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
	#latent transform
	f_params = tf.placeholder(tf.float32, [opt['mb_size'],2,2], name='f_params')
	#transform for validation
	fs_params = tf.placeholder(tf.float32, [1,2,2], name='fs_params')
	lr = tf.placeholder(tf.float32, [], name='lr')
	
	# Build the model
	#transform input
	shape_temp = x.get_shape()
	x_initial_transform = transformer(x, t_params_initial, (28,28))
	x_initial_transform.set_shape(shape_temp)
	#build encoder
	reconstruction, latents = single_model_non_siamese(x_initial_transform,
		f_params, t_params=t_params, conv=opt['convolutional'])
	reconstruction = tf.reshape(reconstruction, x.get_shape())
	#transform input corresponding to latents for loss
	reconstruction_transform = transformer(x_initial_transform, t_params, (28,28))

	loss = tf.reduce_mean(tf.reduce_sum(tf.square(reconstruction_transform - reconstruction), axis=(1,2)))

	recon = single_model(xs, fs_params, name_scope='mainModel', conv=opt['convolutional'], t_params=t_params)

	loss_summary = tf.summary.scalar('Loss', loss)
	lr_summary = tf.summary.scalar('LearningRate', lr)
	
	merged = tf.summary.merge_all()
	
	# Build optimizer
	optim = tf.train.AdamOptimizer(lr)
	train_op = optim.minimize(loss, global_step=global_step)
	
	inputs = [x, global_step, t_params_initial, t_params, f_params, lr, xs, fs_params]
	outputs = [loss, merged, recon]
	ops = [train_op]
	
	# Train
	return train(inputs, outputs, ops, opt)

if __name__ == '__main__':
	opt = {}
	main(opt)
