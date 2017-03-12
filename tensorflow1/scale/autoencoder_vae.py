'''Autoencoder'''

import os
import sys
import time
import glob
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

import matplotlib.pyplot as plt

################ DATA #################

#-----------ARGS----------
flags = tf.app.flags
FLAGS = flags.FLAGS
#execution modes
flags.DEFINE_boolean('ANALYSE', False, 'runs model analysis')
flags.DEFINE_integer('eq_dim', -1, 'number of latent units to rotate')
flags.DEFINE_integer('num_latents', 30, 'Dimension of the latent variables')
flags.DEFINE_float('l2_latent_reg', 1e-6, 'Strength of l2 regularisation on latents')
flags.DEFINE_integer('save_step', 50, 'Interval (epoch) for which to save')
flags.DEFINE_boolean('Daniel', False, 'Daniel execution environment')
flags.DEFINE_boolean('Sleepy', False, 'Sleepy execution environment')
##---------------------

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


def checkFolder(dir):
	"""Checks if a folder exists and creates it if not.
	dir: directory
	Returns nothing
	"""
	if not os.path.exists(dir):
		os.makedirs(dir)


def removeAllFilesInDirectory(directory, extension):
	cwd = os.getcwd()
	os.chdir(directory)
	filelist = glob.glob('*' + extension)
	for f in filelist:
		os.remove(f)
	os.chdir(cwd)


############## MODEL ####################
def transformer_layer(x, t_params, imsh):
	"""Spatial transformer wtih shapes sets"""
	xsh = x.get_shape()
	x_in = transformer(x, t_params, imsh)
	x_in.set_shape(xsh)
	return x_in
	

def flatten(input):
	s = input.get_shape().as_list()
	num_params = 1
	for i in range(1, len(s)): #ignore batch size
		num_params *= s[i]
	return tf.reshape(input, [s[0], num_params])


def linear(x, shape, name='0', bias_init=0.01):
	"""Basic linear matmul layer"""
	He_initializer = tf.contrib.layers.variance_scaling_initializer()
	W = tf.get_variable(name+'_W', shape=shape, initializer=He_initializer)
	z = tf.matmul(x, W, name='mul'+str(name))
	return bias_add(z, shape[1], bias_init=bias_init, name=name)


def bias_add(x, nc, bias_init=0.01, name='0'):
	const_initializer = tf.constant_initializer(value=bias_init)
	b = tf.get_variable(name+'_b', shape=nc, initializer=const_initializer)
	return tf.nn.bias_add(x, b)


def autoencoder(x, f_params, is_training, reuse=False):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	with tf.variable_scope('mainModel', reuse=reuse) as scope:
		x = tf.reshape(x, tf.stack([xsh[0],784]))
		with tf.variable_scope("encoder", reuse=reuse) as scope:
			mu, sigma = encoder(x, is_training, reuse=reuse)
			z = sampler(mu, sigma, sample=False) #(not reuse))
		with tf.variable_scope("feature_transformer", reuse=reuse) as scope:
			matrix_shape = [xsh[0], z.get_shape()[1]]
			z = el.feature_transform_matrix_n(z, matrix_shape, f_params)
		with tf.variable_scope("decoder", reuse=reuse) as scope:
			r = decoder(z, is_training, reuse=reuse)
	return r, z, mu, sigma


def sampler(mu, sigma, sample=True):
	if sample:
		z = mu + sigma*tf.random_normal(mu.get_shape())
	else:
		z = mu
	return z


def encoder(x, is_training, reuse=False):
	"""Encoder MLP"""
	l1 = linear(x, [784,512], name='e0')
	l1 = bn2d(l1, is_training, reuse=reuse, name='b1')
	l2 = linear(tf.nn.elu(l1), [512,512], name='e1')
	l2 = bn2d(l2, is_training, reuse=reuse, name='b2')
	mu = linear(tf.nn.elu(l2), [512,FLAGS.num_latents], name='mu')
	rho = linear(tf.nn.elu(l2), [512,FLAGS.num_latents], name='rho')
	sigma = tf.nn.softplus(rho)
	return mu, sigma


def decoder(z, is_training, reuse=False):
	"""Encoder MLP"""
	l2 = linear(z, [z.get_shape()[1], 512], name='d2')
	l2 = bn2d(l2, is_training, reuse=reuse, name='b2')
	l1 = linear(tf.nn.elu(l2), [512,512], name='d1')
	l1 = bn2d(l1, is_training, reuse=reuse, name='b1')
	return linear(tf.nn.elu(l1), [512,784], name='d0')


def bernoulli_xentropy(x, recon_test):
	"""Cross-entropy for Bernoulli variables"""
	x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_test)
	return tf.reduce_mean(tf.reduce_sum(x_entropy, axis=(1,2)))


def gaussian_kl(mu, sigma):
	"""Reverse KL-divergence from Gaussian variational to unit Gaussian prior"""
	per_neuron_kl = 1. + 2.*tf.log(sigma) - tf.square(mu) - tf.square(sigma)
	return -0.5*tf.reduce_mean(tf.reduce_sum(per_neuron_kl, axis=1))
	

def random_rss(mb_size, imsh, fv=None):
	"""Random rotation, scalex and scaley"""
	t_params = []
	f_params = []
	def scale_(t, a, b):
		t_ = t / np.pi
		return (b-a)*t_ + a
	
	if fv is None:
		fv = np.pi*np.random.rand(mb_size, 3)
		fv[:,0] = 2.*fv[:,0]

	for i in xrange(mb_size):
		# Anisotropically scaled and rotated
		rot = np.array([[np.cos(fv[i,0]), -np.sin(fv[i,0])],
							[np.sin(fv[i,0]), np.cos(fv[i,0])]])
		scale = np.array([[scale_(fv[i,1],0.8,1.8),0.],[0., scale_(fv[i,2],0.8,1.8)]])
		transform = np.dot(scale, rot)
		# Compute transformer matrices
		t_params.append(el.get_t_transform_n(transform, (imsh[0],imsh[1])))
		f_params.append(el.get_f_transform_n(fv[i,:]))
	return np.vstack(t_params), np.stack(f_params, axis=0)


##### SPECIAL FUNCTIONS #####
def bn2d(X, train_phase, decay=0.99, name='batchNorm', reuse=False):
	"""Batch normalization module.
	
	X: tf tensor
	train_phase: boolean flag True: training mode, False: test mode
	decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
	name: (default batchNorm)
	
	Source: bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
	batch-normalization-in-tensorflow"""
	n_out = X.get_shape().as_list()[1]
	
	beta = tf.get_variable('beta_'+name, dtype=tf.float32, shape=n_out,
								  initializer=tf.constant_initializer(0.0))
	gamma = tf.get_variable('gamma_'+name, dtype=tf.float32, shape=n_out,
									initializer=tf.constant_initializer(1.0))
	pop_mean = tf.get_variable('pop_mean_'+name, dtype=tf.float32,
										shape=n_out, trainable=False)
	pop_var = tf.get_variable('pop_var_'+name, dtype=tf.float32,
									  shape=n_out, trainable=False)
	batch_mean, batch_var = tf.nn.moments(X, [0], name='moments_'+name)
	
	if not reuse:
		ema = tf.train.ExponentialMovingAverage(decay=decay)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
			pop_var_op = tf.assign(pop_var, ema.average(batch_var))
	
			with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		
		mean, var = tf.cond(train_phase, mean_var_with_update,
					lambda: (pop_mean, pop_var))
	else:
		mean, var = tf.cond(train_phase, lambda: (batch_mean, batch_var),
				lambda: (pop_mean, pop_var))
		
	return tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)


############################################
def train(inputs, outputs, ops, opt, data):
	"""Training loop"""
	# Unpack inputs, outputs and ops
	x, global_step, t_params_in, t_params, f_params, lr, xs, f_params_val, t_params_val, is_training = inputs
	loss, merged, recon_test = outputs
	train_op = ops
	
	# For checkpoints
	gs = 0
	start = time.time()
	n_train = data['X']['train'].shape[0]
	n_valid = data['X']['valid'].shape[0]
	n_test = data['X']['test'].shape[0]
	
	saver = tf.train.Saver()
	sess = tf.Session()

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
			tp_in, fp_in = el.random_transform(opt['mb_size'], opt['im_size'])
			#tp, fp = el.random_transform(opt['mb_size'], opt['im_size'])
			tp, fp = random_rss(opt['mb_size'], opt['im_size'])
			ops = [global_step, loss, merged, train_op]
			feed_dict = {x: data['X']['train'][mb,...],
								t_params: tp,
								f_params: fp,
								t_params_in: tp_in,
								lr: current_lr,
								is_training: True}
			gs, l, summary, __ = sess.run(ops, feed_dict=feed_dict)
			train_loss += l
			# Summary writers
			train_writer.add_summary(summary, gs)
		train_loss /= (i+1.)
		print('[{:03d}]: {:03f}'.format(epoch, train_loss))
		
		# Save model
		if epoch % FLAGS.save_step == 0:
			path = saver.save(sess, opt['save_path'], epoch)
			print('Saved model to ' + path)
		
		# Validation
		if epoch % 10 == 0:
			Recon = []
			max_angles = 20
			#pick a random initial transformation
			tp_in, fp_in = el.random_transform(opt['mb_size'], opt['im_size'])
			fv = np.linspace(0., np.pi, num=max_angles)
			for j in xrange(max_angles):
				sample = data['X']['valid'][np.newaxis,np.random.randint(5000),...]
				r0 = np.random.rand() > 0.5
				r1 = np.random.rand() > 0.5
				r2 = np.random.rand() > 0.5
				fv_ = np.vstack([2.*r0*fv, r1*fv, r2*fv]).T
				
				Recon.append(np.reshape(sample, (1,784)))
				for i in xrange(max_angles):
					tp, fp = random_rss(1, opt['im_size'], fv_[np.newaxis,i,:])
					ops = recon_test
					feed_dict = {xs: sample,
									 f_params_val: fp,
									 t_params_val: tp,
									 t_params_in: tp_in,
									 is_training: False}
					y= sess.run(ops, feed_dict=feed_dict)
					Recon.append(sess.run(ops, feed_dict=feed_dict))
			
			samples_ = np.reshape(Recon, (-1,28,28))
			
			ns = max_angles
			sh = 28
			tile_image = np.zeros((ns*sh,(ns+1)*sh))
			for j in xrange((ns+1)*ns):
				m = sh*(j/(ns+1)) 
				n = sh*(j%(ns+1))
				tile_image[m:m+sh,n:n+sh] = 1.-samples_[j,...]
			save_name = './samples/vae/image_%04d.png' % epoch
			skio.imsave(save_name, tile_image)


def main(_):
	opt = {}
	"""Main loop"""
	tf.reset_default_graph()
	if FLAGS.Daniel:
		print('Hello Daniel!')
		opt['root'] = '/home/daniel'
		dir_ = opt['root'] + '/Code/harmonicConvolutions/tensorflow1/scale'
	elif FLAGS.Sleepy:
		print('Hello dworrall!')
		opt['root'] = '/home/dworrall'
		dir_ = opt['root'] + '/Code/harmonicConvolutions/tensorflow1/scale'
	else:
		opt['root'] = '/home/sgarbin'
		dir_ = opt['root'] + '/Projects/harmonicConvolutions/tensorflow1/scale'
	opt['mb_size'] = 128
	opt['n_channels'] = 10
	opt['n_epochs'] = 2000
	opt['lr_schedule'] = [50, 75]
	opt['lr'] = 1e-3
	opt['im_size'] = (28,28)
	opt['train_size'] = 55000
	opt['equivariant_weight'] = 1 
	flag = 'vae'
	opt['summary_path'] = dir_ + '/summaries/autotrain_{:s}'.format(flag)
	opt['save_path'] = dir_ + '/checkpoints/autotrain_{:s}/model.ckpt'.format(flag)

	#check and clear directories
	checkFolder(opt['summary_path'])
	checkFolder(opt['save_path'])
	removeAllFilesInDirectory(opt['summary_path'], '.*')
	removeAllFilesInDirectory(opt['save_path'], '.*')
	
	# Load data
	data = load_data()
	data['X']['train'] = data['X']['train'][:opt['train_size'],:]
	data['Y']['train'] = data['Y']['train'][:opt['train_size'],:]

	# Placeholders
	x = tf.placeholder(tf.float32, [opt['mb_size'],28,28,1], name='x')
	xs = tf.placeholder(tf.float32, [1,28,28,1], name='xs')
	t_params_in = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params_in')
	t_params = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
	f_params = tf.placeholder(tf.float32, [opt['mb_size'],6,6], name='f_params')
	t_params_val = tf.placeholder(tf.float32, [1,6], name='t_params_val') 
	f_params_val = tf.placeholder(tf.float32, [1,6,6], name='f_params_val') 
	global_step = tf.Variable(0, name='global_step', trainable=False)
	lr = tf.placeholder(tf.float32, [], name='lr')
	is_training = tf.placeholder(tf.bool, [], name='is_training')
	
	# Build the training model
	x_in = transformer_layer(x, t_params_in, opt['im_size'])
	target = transformer_layer(x_in, t_params, opt['im_size'])
	recon, latents, mu, sigma = autoencoder(x_in, f_params, is_training)
	recon = tf.reshape(recon, x.get_shape())
	# Test model
	recon_test_, __, __, __ = autoencoder(xs, f_params_val, is_training, reuse=True)
	recon_test = tf.nn.sigmoid(recon_test_)
	
	# LOSS
	# KL-divergence of posterior from prior
	kl_loss = gaussian_kl(mu, sigma)
	# Negative log-likelihood
	nll = bernoulli_xentropy(tf.to_float(target > 0.5), recon)
	loss = nll #+ kl_loss
	
	# Summaries
	tf.summary.scalar('Loss', loss)
	tf.summary.scalar('Loss_KL', kl_loss)
	tf.summary.scalar('Loss_Img', nll)
	tf.summary.scalar('LearningRate', lr)
	merged = tf.summary.merge_all()
	
	# Build optimizer
	optim = tf.train.AdamOptimizer(lr)
	train_op = optim.minimize(loss, global_step=global_step)
	
	# Set inputs, outputs, and training ops
	inputs = [x, global_step, t_params_in, t_params, f_params, lr, xs, f_params_val, t_params_val, is_training]
	outputs = [loss, merged, recon_test]
	ops = [train_op]
	
	# Train
	return train(inputs, outputs, ops, opt, data)

if __name__ == '__main__':
	tf.app.run()