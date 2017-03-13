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
def dot_product(x, y):
	return tf.reduce_sum(tf.multiply(x,y), axis=1)

def cosine_distance(x, y):
	return tf.reduce_mean(dot_product(x, y) / (dot_product(x, x) * dot_product(y, y)))

def convolutional(x, conv_size, num_filters, stride=1, name='c0',
		bias_init=0.01, padding='SAME', non_linear_func=tf.nn.relu):
	w = tf.get_variable(name + '_conv_w', [conv_size, conv_size, x.get_shape()[3], num_filters])
	b = tf.get_variable(name + '_conv_b', [num_filters])
	result = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding, name=name+'conv')
	return non_linear_func(bias_add(result, num_filters, name=name))

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

'''
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


def encoder_conv(x, n_mid=10, n_layers=2):
	"""Encoder MLP"""
	x = convolutional(x, 3, n_mid, name='e0')
	for i in xrange(1,n_layers-2):
		x = convolutional(x, 3, n_mid*2**(i/2), stride=1+(i%2==0), name='e{:d}'.format(i))
	return convolutional(x, 3, n_mid*2**((n_layers-1)/2), non_linear_func=(lambda x: x), name='e_out')


def decoder_conv(z, n_mid=10, n_layers=2):
	"""Encoder MLP"""
	bs = z.get_shape()[0]
	out_shape = [bs,28/(2**((n_layers-1)/2)),28/(2**((n_layers-1)/2)),n_mid*2**((n_layers-1)/2)]
	z = deconvolutional_deepmind(z, 3, out_shape, name='d_in')
	for i in xrange(n_layers-3,0,-1):
		out_shape = [bs,28/(2**(i/2)),28/(2**(i/2)),n_mid*2**(i/2)]
		z = deconvolutional_deepmind(z, 3, out_shape, name='d{:d}'.format(i))
	return deconvolutional_deepmind(z, 3, [bs,28,28,1], name='d0')


def classifier(x, n_mid=128, n_hid=128, n_layers=2):
	x = linear(x, [n_hid,n_mid], name='c0')
	for i in xrange(1,n_layers-2):
		x = linear(x, [n_mid,n_mid], name='c{:d}'.format(i))
	return linear(tf.nn.relu(x), [n_mid,10], name='c_out')


def classifier_conv(x, n_mid=10, layer_in=2, n_layers=2):
	x = convolutional(x, 3, n_mid*2**(layer_in/2), stride=1+(layer_in%2==0), name='c0')
	for i in xrange(1,n_layers-2):
		x = convolutional(x, 3, n_mid*2**((layer_in+i)/2), stride=1+((layer_in+i)%2==0), name='c{:d}'.format(i))
	x = convolutional(x, 3, 10, non_linear_func=(lambda x: x), name='c_out')
	return tf.reduce_mean(x, axis=(1,2))


######################################################

def train(inputs, outputs, ops, opt):
	"""Training loop"""
	x, global_step, t_params_initial, t_params, f_params, lr, labels, vacc, fs_params, ts_params, xs, t_params_initial = inputs
	loss, merged, acc, vacc_summary = outputs
	train_op, recon = ops
	
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
								 labels: data['Y']['train'][mb]}
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
			train_acc /= (i+1)
			
			# Validation
			valid_acc = 0.
			mb_list = random_sampler(n_valid, opt, random=False)
			for i, mb in enumerate(mb_list):
				ops = acc
				tp = np.asarray([[1.,0.,0.,0.,1.,0.]])*np.ones([opt['mb_size'],1])
				feed_dict = {x: data['X']['valid'][mb,...],
								 labels: data['Y']['valid'][mb],
								 t_params_initial: tp}
				c = sess.run(ops, feed_dict=feed_dict)
				valid_acc += c
				# Summary writers
			valid_acc /= (i+1)
			vs = sess.run(vacc_summary, feed_dict={vacc: valid_acc})
			train_writer.add_summary(vs, gs)
			print('[{:03d}]: Train Loss {:03f}, Train Acc {:03f}, Valid Acc {:03f}'.format(epoch, train_loss, train_acc, valid_acc))
			
			'''
			# Try reconstruction
			if epoch % 10 == 0:
				Recon = []
				sample = data['X']['valid'][np.newaxis,np.random.randint(n_valid),...]
				
				max_angles = 20*20
				#pick a random initial transformation
				tp_init, fp_init = el.random_transform(opt['mb_size'], opt['im_size'])
				for i in xrange(max_angles):
					#fp = el.get_f_transform(2.*np.pi*i/(1.*max_angles))[np.newaxis,:,:]
					tp, fp = el.random_transform_theta(1, opt['im_size'],
						2.*np.pi*i/(1.*max_angles)) # last arg is theta
					#print fp
					ops = recon
					feed_dict = {xs: sample,
									 fs_params: fp,
									 ts_params: tp,
									 t_params_initial: tp_init}
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
			'''
		
		# Test
		test_acc = 0.
		mb_list = random_sampler(n_test, opt, random=False)
		for i, mb in enumerate(mb_list):
			ops = acc
			tp = np.asarray([[1.,0.,0.,0.,1.,0.]])*np.ones([opt['mb_size'],1])
			feed_dict = {x: data['X']['test'][mb,...],
							 labels: data['Y']['test'][mb],
							 t_params_initial: tp}
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
		opt['n_layers'] = 4
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
	
	# Build the model
	# Input -- initial transformation
	shape_temp = x.get_shape()
	x_initial_transform = transformer(x, t_params_initial, (28,28))
	x_initial_transform.set_shape(shape_temp)
	
	# Autoencoder
	'''
	reconstruction, latents = single_model_non_siamese(x_initial_transform,
		f_params, t_params=t_params, n_mid=opt['n_mid'], n_hid=opt['n_hid'], n_layers=opt['n_layers'])
	reconstruction = tf.reshape(reconstruction, x.get_shape())
	'''
	reconstruction, latents = single_model_non_siamese_conv(x_initial_transform,
		f_params, t_params, n_mid=opt['n_mid'], n_layers=opt['n_layers'])
	reconstruction = tf.reshape(reconstruction, x.get_shape())
	
	# Branch 2 -- transform input data
	reconstruction_transform = transformer(x_initial_transform, t_params, (28,28))
	reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reconstruction_transform - reconstruction), axis=(1,2)))
	
	# Classification
	#logits = classifier(latents, n_mid=opt['n_mid_class'], n_hid=opt['n_hid'], n_layers=opt['n_layers_class'])
	logits = classifier_conv(latents, n_mid=opt['n_mid_class'], layer_in=opt['n_layers'], n_layers=opt['n_layers_class'])
	classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	loss = (1-opt['equivariant_weight'])*tf.reduce_mean(classification_loss) + opt['equivariant_weight']*reconstruction_loss
	acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1), labels)))

	# Print reconstructions
	#recon = single_model(xs, fs_params, name_scope='mainModel', t_params=ts_params)
	recon = single_model_conv(xs, fs_params, ts_params, n_mid=opt['n_mid'], n_layers=opt['n_layers'], name_scope='mainModel')
	
	# Summaries
	loss_summary = tf.summary.scalar('Loss', loss)
	tacc_summary = tf.summary.scalar('Training Accuracy', acc)
	lr_summary = tf.summary.scalar('LearningRate', lr)
	merged = tf.summary.merge_all()
	
	vacc_summary = tf.summary.scalar('Validation Accuracy', vacc)
	
	# Build optimizer
	optim = tf.train.AdamOptimizer(lr)
	train_op = optim.minimize(loss, global_step=global_step)
	
	inputs = [x, global_step, t_params_initial, t_params, f_params, lr, labels, vacc, fs_params, ts_params, xs, t_params_initial]
	outputs = [loss, merged, acc, vacc_summary]
	ops = [train_op, recon]
	
	# Train
	return train(inputs, outputs, ops, opt)

if __name__ == '__main__':
	main()





















