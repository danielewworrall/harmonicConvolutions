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
	train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	test = np.load('./data/mnist_rotation_new/rotated_test.npz')

	data = {}
	data['X'] = {'train': np.reshape(train['x'], (-1,28,28,1)),
					 'valid': np.reshape(valid['x'], (-1,28,28,1)),
					 'test': np.reshape(test['x'], (-1,28,28,1))}
	data['Y'] = {'train': train['y'],
					 'valid': valid['y'],
					 'test': test['y']}
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
def leaky_relu(x, leak=0.1):
	"""The leaky ReLU nonlinearity"""
	return tf.nn.relu(x) + tf.nn.relu(-leak*x)


def conv(x, shape, stride=1, name='0', bias_init=0.01, padding='SAME'):
	with tf.variable_scope('conv') as scope:
		He_initializer = tf.contrib.layers.variance_scaling_initializer()
		W = tf.get_variable(name+'_W', shape=shape, initializer=He_initializer)
		z = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding, name=name+'conv')
		return bias_add(z, shape[3], bias_init=bias_init, name=name)


def deconv(x, W_shape, out_shape, stride=1, name='0', bias_init=0.01, padding='SAME'):
	with tf.variable_scope('deconv') as scope:
		# Resize convolution a la Google Brain
		y = tf.image.resize_images(x, out_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		# Perform convolution, zero padding implicit
		return conv(y, W_shape, stride=stride, name=name, bias_init=bias_init, padding=padding)


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

'''
def single_model_conv(x, f_params, t_params, n_mid, n_layers, name_scope='siamese'):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope(name_scope, reuse=True) as scope:
		with tf.variable_scope("Encoder", reuse=True) as scope:
			z = encoder_conv(x, n_mid=n_mid, n_layers=n_layers)
		z_rot = el.transform_features(z, t_params, f_params)
		with tf.variable_scope("Decoder", reuse=True) as scope:
			r = decoder_conv(z_rot, n_mid=n_mid, n_layers=n_layers)
	return r


def single_model_non_siamese(x, f_params, conv=False, t_params=[], n_mid=512,
									  n_hid=128, n_layers=2):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('mainModel', reuse=False) as scope:
		# Basic branch
		x = tf.reshape(x, tf.stack([xsh[0],784]))
		with tf.variable_scope("Encoder", reuse=False) as scope:
			z = encoder(x, n_mid=n_mid, n_hid=n_hid, n_layers=n_layers)
		z_rot = el.feature_space_transform2d(z, [xsh[0], n_hid], f_params)
		with tf.variable_scope("Decoder", reuse=False) as scope:
			r = decoder(z_rot, n_mid=n_mid, n_hid=n_hid, n_layers=n_layers)
	return r, z


def single_model_non_siamese_conv(x, f_params, t_params, n_mid=10, n_layers=2):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('mainModel', reuse=False) as scope:
		# Basic branch
		#x = tf.reshape(x, tf.stack([xsh[0],784]))
		with tf.variable_scope("Encoder", reuse=False) as scope:
			z = encoder_conv(x, n_mid=n_mid, n_layers=n_layers)
		z_rot = el.transform_features(z, t_params, f_params)
		with tf.variable_scope("Decoder", reuse=False) as scope:
			r = decoder_conv(z_rot, n_mid=n_mid, n_layers=n_layers)
	return r, z
'''

def autoencoder(x, f_params, t_params, is_training, opt, reuse=False):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	with tf.variable_scope('mainModel', reuse=reuse) as scope:
		with tf.variable_scope("encoder", reuse=reuse) as scope:
			z = encoder_conv(x, is_training, opt, reuse=reuse)
		with tf.variable_scope("feature_transformer", reuse=reuse) as scope:
			z_ = el.transform_features_tensor(z, t_params, f_params)
		with tf.variable_scope("decoder", reuse=reuse) as scope:
			r = decoder_conv(z_, is_training, opt, reuse=reuse)
	return r, z


def encoder(x, n_mid=512, n_hid=128, n_layers=2):
	"""Encoder MLP"""
	x = linear(x, [784,n_mid], name='e0')
	for i in xrange(1,n_layers-2):
		x = linear(x, [n_mid,n_mid], name='e{:d}'.format(i))
	return linear(tf.nn.relu(x), [n_mid,n_hid], name='e_hid')


def decoder(z, n_mid=512, n_hid=128, n_layers=2):
	"""Encoder MLP"""
	z = linear(z, [n_hid,n_mid], name='d0')
	for i in xrange(1,n_layers-2):
		z = linear(z, [n_mid,n_mid], name='d{:d}'.format(i))
	return tf.nn.sigmoid(linear(tf.nn.relu(z), [n_mid,784], name='d_out'))


def encoder_conv(x, is_training, opt, reuse=False):
	"""Encoder CNN"""
	n_mid = opt['n_mid']
	n_layers = opt['n_layers']
	
	with tf.variable_scope('encoder_0', reuse=reuse) as scope:
		x = conv(x, [3,3,1,n_mid], name='e0', padding='VALID')
		x = bn4d(x, is_training, name='bn0', reuse=reuse)
		x = leaky_relu(x)
		
	nl_ = n_mid
	for i in xrange(1,n_layers-1):
		with tf.variable_scope('encoder_{:d}'.format(i), reuse=reuse) as scope:
			nl = n_mid*2**(i/2)
			x = conv(x, [3,3,nl_,nl], stride=1+(i==2), name='e{:d}'.format(i), padding='VALID')
			x = bn4d(x, is_training, name='bn_e{:d}'.format(i), reuse=reuse)
			x = leaky_relu(x)
			nl_ = nl
	
	with tf.variable_scope('Encoder_out', reuse=reuse) as scope:
		return conv(x, [3,3,nl_,n_mid*2**((n_layers-1)/2)], name='e_out', padding='VALID')


def decoder_conv(z, is_training, opt, reuse=False):
	"""Decoder CNN"""
	n_mid = opt['n_mid']
	n_layers = opt['n_layers_deconv']
	
	zsh = z.get_shape().as_list()
	bs = zsh[0]
	k = zsh[1]
	nl_mult = np.exp(np.log(zsh[3]/(1.*n_mid)) / (n_layers-1))

	with tf.variable_scope('decoder_in', reuse=reuse) as scope:
		out_shape = get_outshape(k,28,n_layers,n_layers)
		nl = int(np.power(nl_mult, n_layers-1)*n_mid)
		z = deconv(z, [5,5,zsh[3],nl], out_shape, name='d_in')
		z = bn4d(z, is_training, name='bn_d_in', reuse=reuse)

	nl_ = nl
	for i in xrange(n_layers-2,0,-1):
		print nl
		with tf.variable_scope('decoder_{:d}'.format(i), reuse=reuse) as scope:
			out_shape = get_outshape(k,28,i+1,n_layers)
			nl = int(np.power(nl_mult,i)*n_mid)
			z = deconv(leaky_relu(z),  [5,5,nl_,nl], out_shape, name='d{:d}'.format(i))
			z = bn4d(z, is_training, name='bn_d{:d}'.format(i), reuse=reuse)
			z = leaky_relu(z)
			nl_ = nl
	print nl
	with tf.variable_scope('decoder_1', reuse=reuse) as scope:
		return deconv(leaky_relu(z), [3,3,nl_,1], [28,28], name='d0')


def get_outshape(input_size, output_size, layer_num, n_layers):
	downsample_factor = np.exp(np.log((1.*input_size)/output_size)/n_layers)
	size = output_size*(downsample_factor**layer_num)
	return [int(size),int(size)]


def classifier(x, is_training, opt):
	n_layers = opt['n_layers']
	sigma = opt['dropout']*tf.to_float(is_training)
	# GAP
	x = tf.reduce_mean(x, axis=(1,2))
	# Invariant rep
	xsh = x.get_shape().as_list()
	x = tf.reshape(x, [-1,xsh[1]/2,2])
	x = tf.reduce_sum(tf.square(x), axis=2)
	xsh = x.get_shape().as_list()
	# Dropout
	x = x*(1. + sigma*tf.random_normal(x.get_shape()))
	x = linear(x, [xsh[1],xsh[1]], name='c0')
	for i in xrange(1,n_layers-2):
		x = x*(1. + sigma*tf.random_normal(x.get_shape()))
		x = linear(x, [xsh[1],xsh[1]], name='c{:d}'.format(i))
	x = x*(1. + sigma*tf.random_normal(x.get_shape()))
	return linear(leaky_relu(x), [xsh[1],10], name='c_out')


def classifier_conv(x, n_mid=10, layer_in=2, n_layers=2):
	x = convolutional(x, 3, n_mid*2**(layer_in/2), stride=1+(layer_in%2==0), name='c0')
	for i in xrange(1,n_layers-2):
		x = convolutional(x, 3, n_mid*2**((layer_in+i)/2), stride=1+((layer_in+i)%2==0), name='c{:d}'.format(i))
	x = convolutional(x, 3, 10, non_linear_func=(lambda x: x), name='c_out')
	return tf.reduce_mean(x, axis=(1,2))


def bn4d(X, train_phase, decay=0.99, name='batchNorm', reuse=False):
	"""Batch normalization module.
	
	X: tf tensor
	train_phase: boolean flag True: training mode, False: test mode
	decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
	name: (default batchNorm)
	
	Source: bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
	batch-normalization-in-tensorflow"""
	with tf.variable_scope(name, reuse=reuse) as scope:
		n_out = X.get_shape().as_list()[3]
		
		beta = tf.get_variable('beta', dtype=tf.float32, shape=n_out,
									  initializer=tf.constant_initializer(0.0))
		gamma = tf.get_variable('gamma', dtype=tf.float32, shape=n_out,
										initializer=tf.constant_initializer(1.0))
		pop_mean = tf.get_variable('pop_mean', dtype=tf.float32,
											shape=n_out, trainable=False)
		pop_var = tf.get_variable('pop_var', dtype=tf.float32,
										  shape=n_out, trainable=False)
		batch_mean, batch_var = tf.nn.moments(X, [0,1,2], name='moments_'+name)
		
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

######################################################

def train(inputs, outputs, ops, opt):
	"""Training loop"""
	x, global_step, t_params_initial, t_params, f_params, lr, labels, vacc, fs_params, ts_params, xs, t_params_initial, is_training, annealer = inputs
	loss, merged, acc, vacc_summary = outputs
	train_op = ops

	# For checkpoints
	saver = tf.train.Saver()
	gs = 0
	start = time.time()
	
	data = load_data()
	data['X']['train'] = data['X']['train'][:opt['train_size'],:]
	data['Y']['train'] = data['Y']['train'][:opt['train_size']]
	n_train = data['X']['train'].shape[0]
	n_valid = data['X']['valid'].shape[0]
	n_test = data['X']['test'].shape[0]

	with tf.Session() as sess:
		# Threading and queueing
		#coord = tf.train.Coordinator()
		#threads = tf.train.start_queue_runners(coord=coord)
		
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
			current_anneal = np.minimum(0.95,0.0025*epoch)
			for i, mb in enumerate(mb_list):
				#initial random transform
				tp_init, fp_init = el.random_transform(opt['mb_size'], opt['im_size'])

				tp, fp = el.random_transform(opt['mb_size'], opt['im_size'])
				ops = [global_step, loss, merged, train_op, acc]
				feed_dict = {x: data['X']['train'][mb,...],
								 t_params: tp,
								 f_params: fp,
								 t_params_initial: tp_init,
								 lr: current_lr,
								 labels: data['Y']['train'][mb],
								 is_training: True,
								 annealer: current_anneal}
				gs, l, summary, __, c = sess.run(ops, feed_dict=feed_dict)
				train_loss += l
				train_acc += c
				if np.isnan(l):
					print('NaN encountered')
					sess.close()
					return -1
				# Summary writers
				train_writer.add_summary(summary, gs)
			train_loss /= (i+1.)
			train_acc /= (i+1.)
			
			if epoch > 100:
				if train_acc < 0.50:
					print('Not worth it: {:03f}'.format(train_acc))
					sess.close()
					return -1
			
			# Validation
			valid_acc = 0.
			mb_list = random_sampler(n_valid, opt, random=False)
			for i, mb in enumerate(mb_list):
				ops = acc
				tp = np.asarray([[1.,0.,0.,0.,1.,0.]])*np.ones([opt['mb_size'],1])
				feed_dict = {x: data['X']['valid'][mb,...],
								 labels: data['Y']['valid'][mb],
								 t_params_initial: tp,
								 is_training: False}
				c = sess.run(ops, feed_dict=feed_dict)
				valid_acc += c
				# Summary writers
			valid_acc /= (i+1)
			vs = sess.run(vacc_summary, feed_dict={vacc: valid_acc})
			train_writer.add_summary(vs, gs)
			print('[{:03d}]: Train Loss {:03f}, Train Acc {:03f}, Valid Acc {:03f}, Anneal: {:03f}'.format(epoch, train_loss, train_acc, valid_acc, current_anneal))

		
		# Test
		test_acc = 0.
		mb_list = random_sampler(n_test, opt, random=False)
		for i, mb in enumerate(mb_list):
			ops = acc
			tp = np.asarray([[1.,0.,0.,0.,1.,0.]])*np.ones([opt['mb_size'],1])
			feed_dict = {x: data['X']['test'][mb,...],
							 labels: data['Y']['test'][mb],
							 t_params_initial: tp,
							 is_training: False}
			c = sess.run(ops, feed_dict=feed_dict)
			test_acc += c
			# Summary writers
		test_acc /= (i+1)
		print('[{:03d}]: Test Acc {:03f}'.format(epoch, test_acc))
	return test_acc

def main(opt=None):
	"""Main loop"""
	tf.reset_default_graph()
	
	if opt is None:
		opt = {}
		opt['root'] = '/home/dworrall'
		dir_ = opt['root'] + '/Code/harmonicConvolutions/tensorflow1/scale'
		
		opt['mb_size'] = 128
		opt['n_channels'] = 10
		opt['n_epochs'] = 1000
		opt['lr_schedule'] = [500, 750]
		opt['lr'] = 1e-3
		opt['save_step'] = 10
		opt['im_size'] = (28,28)
		opt['train_size'] = 10000
		opt['equivariant_weight'] = 0.1
		
		opt['n_mid'] = 10
		opt['n_hid'] = 10
		opt['n_layers'] = 8
		opt['n_layers_deconv'] = 4
		opt['n_mid_class'] = 10
		opt['n_layers_class'] = 4
	
		flag = 'bn'
		opt['summary_path'] = dir_ + '/summaries/rot_mnist_{:.0e}_{:s}'.format(opt['equivariant_weight'], flag)
		opt['save_path'] = dir_ + '/checkpoints/rot_mnist_{:.0e}_{:s}/model.ckpt'.format(opt['equivariant_weight'], flag)
		opt['loss_type_image'] = 'l2'

	# Construct input graph
	# Input image
	x = tf.placeholder(tf.float32, [opt['mb_size'],28,28,1], name='x')
	labels = tf.placeholder(tf.int64, [opt['mb_size']], name='labels')
	vacc = tf.placeholder(tf.float32, [], name='vacc')
	# Validation input
	#xs = tf.placeholder(tf.float32, [1,28,28,1], name='xs')
	# Define variables
	global_step = tf.Variable(0, name='global_step', trainable=False)
	#initial transformation
	t_params_initial = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
	#transform corresponding to latents
	t_params = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
	#latent transform
	f_params = tf.placeholder(tf.float32, [opt['mb_size'],2,2], name='f_params')
	#transform for validation
	xs = tf.placeholder(tf.float32, [1,28,28,1], name='xs')
	fs_params = tf.placeholder(tf.float32, [1,2,2], name='fs_params')
	ts_params = tf.placeholder(tf.float32, [1,6], name='ts_params')
	ts_params_initial = tf.placeholder(tf.float32, [1,6], name='ts_param_initial') 
	#fs_params = tf.placeholder(tf.float32, [1,2,2], name='fs_params') #latents
	lr = tf.placeholder(tf.float32, [], name='lr')
	is_training = tf.placeholder(tf.bool, [], name='is_training')
	annealer = tf.placeholder(tf.float32, [], name='annealer')
	
	# Build the model
	# Input -- initial transformation
	shape_temp = x.get_shape()
	x_initial_transform = transformer(x, t_params_initial, (28,28))
	x_initial_transform.set_shape(shape_temp)
	
	# Autoencoder
	reconstruction, latents = autoencoder(x_initial_transform, f_params, t_params, is_training, opt)
	reconstruction = tf.reshape(reconstruction, x.get_shape())
	
	# Branch 2 -- transform input data
	reconstruction_transform = transformer(x_initial_transform, t_params, (28,28))
	recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reconstruction_transform - reconstruction), axis=(1,2)))
	
	# Classification
	logits = classifier(latents, is_training, opt)
	#logits = classifier_conv(latents, n_mid=opt['n_mid_class'], layer_in=opt['n_layers'], n_layers=opt['n_layers_class'])
	class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	
	loss = annealer*(1-opt['equivariant_weight'])*tf.reduce_mean(class_loss) + (1-annealer)*opt['equivariant_weight']*recon_loss
	acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1), labels)))
	
	# Summaries
	loss_summary = tf.summary.scalar('Loss', loss)
	tacc_summary = tf.summary.scalar('Training Accuracy', acc)
	lr_summary = tf.summary.scalar('LearningRate', lr)
	merged = tf.summary.merge_all()
	
	vacc_summary = tf.summary.scalar('Validation Accuracy', vacc)
	
	# Build optimizer
	optim = tf.train.AdamOptimizer(lr)
	train_op = optim.minimize(loss, global_step=global_step)
	
	inputs = [x, global_step, t_params_initial, t_params, f_params, lr, labels, vacc, fs_params, ts_params, xs, t_params_initial, is_training, annealer]
	outputs = [loss, merged, acc, vacc_summary]
	ops = train_op
	
	# Print num params
	n_params = 0
	for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		n_params += int(np.prod(np.asarray(param.get_shape().as_list())))
	print('Num params: {:d}'.format(n_params))
	
	# Train
	return train(inputs, outputs, ops, opt)

if __name__ == '__main__':
	main()





















