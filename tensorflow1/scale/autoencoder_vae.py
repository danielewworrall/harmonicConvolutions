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
flags.DEFINE_boolean('ANALISE', False, 'runs model analysis')
flags.DEFINE_integer('eq_dim', -1, 'number of latent units to rotate')
flags.DEFINE_integer('num_latents', 32, 'Dimension of the latent variables')
flags.DEFINE_float('l2_latent_reg', 1e-6, 'Strength of l2 regularisation on latents')
flags.DEFINE_integer('save_step', 50, 'Interval (epoch) for which to save')
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
def dot_product(x, y):
	return tf.reduce_sum(tf.multiply(x,y), axis=1)

def cosine_distance(x, y):
	return tf.reduce_mean(dot_product(x, y) / (dot_product(x, x) * dot_product(y, y)))

def convolutional(x, conv_size, num_filters, stride=1, name='c0',
		bias_init=0.01, padding='SAME', non_linear_func=tf.nn.elu):
	w = tf.get_variable(name + '_conv_w', [conv_size, conv_size, x.get_shape()[3], num_filters])
	b = tf.get_variable(name + '_conv_b', [num_filters])
	result = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding, name = name + 'conv')
	return non_linear_func(bias_add(result, num_filters, name=name), name = name + '_conv_nl')

def deconvolutional_deepmind(x, conv_size, out_shape, stride=1, name='c0',
		bias_init=0.01, padding='SAME', non_linear_func=tf.nn.elu):
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
	return tf.reshape(input, [s[0], num_params])

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


def single_model(x, f_params, random_sample, name_scope='siamese', conv=False, t_params=[]):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope(name_scope, reuse=True) as scope:
		# Basic branch
		x = tf.reshape(x, tf.stack([xsh[0],784]))
		with tf.variable_scope("Encoder", reuse=True) as scope:
			mu, log_sigma = encoder(x, conv=conv)
			z = mu + tf.exp(log_sigma) * random_sample
		if conv:
			z = el.transform_features(z, t_params, f_params)
		else:
			if FLAGS.eq_dim == - 1:
				z = el.feature_space_transform2d(z, [xsh[0], z.get_shape()[1]], f_params)
			else:
				eq_part, inv_part = tf.split(z, [FLAGS.eq_dim, z.get_shape().as_list()[1] - FLAGS.eq_dim], axis=1)
				eq_part = el.feature_space_transform2d(eq_part, [xsh[0], FLAGS.eq_dim], f_params)
				z = tf.concat([eq_part, inv_part], 1)
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
			mu, log_sigma = encoder(x, conv=conv)
			z, eps = sampler(mu, log_sigma)
		if conv:
			z = el.transform_features(z, t_params, f_params)
		else:
			if FLAGS.eq_dim == - 1:
				z = el.feature_space_transform2d(z, [xsh[0], z.get_shape()[1]], f_params)
			else:
				eq_part, inv_part = tf.split(z, [FLAGS.eq_dim, z.get_shape().as_list()[1] - FLAGS.eq_dim], axis=1)
				eq_part = el.feature_space_transform2d(eq_part, [xsh[0], FLAGS.eq_dim], f_params)
				z = tf.concat([eq_part, inv_part], 1)
		with tf.variable_scope("Decoder", reuse=False) as scope:
			r = decoder(z, conv=conv)
	return r, z, mu, log_sigma, eps


def autoencoder(x):
	"""Build autoencoder"""
	xsh = x.get_shape().as_list()
	with tf.variable_scope("Encoder") as scope:
		z = encoder(x, conv=False)
	with tf.variable_scope("Decoder") as scope:
		r = decoder(z)
	return z, r

def sampler(mu, log_sigma):
	#reparam trick :)
	eps = tf.random_normal(mu.get_shape())
	return mu + tf.exp(log_sigma) * eps, eps

def encoder(x, conv=False):
	if conv:
		print('encoder activation sizes:')
		stream = tf.reshape(x, [x.get_shape().as_list()[0], 28, 28, 1])
		print(stream.get_shape().as_list())
		stream = convolutional(stream, 3, 16, stride=2, name='c1')
		print(stream.get_shape().as_list())
		stream = convolutional(stream, 3, 32, stride=2, name='c2')
		print(stream.get_shape().as_list())
		stream = convolutional(stream, 3, 64, stride=2, name='c3', non_linear_func=tf.identity)
		print(stream.get_shape().as_list())
		return stream
	else:
		"""Encoder MLP"""
		l1 = linear(x, [784,512], name='1')
		#l2 = linear(tf.nn.elu(l1), [500,500], name='2')
		#l2 = linear(tf.nn.elu(l2), [500,500], name='2b')
		#l2 = linear(tf.nn.elu(l2), [500,500], name='2c')
		#l3 = linear(tf.nn.elu(l2), [500,256], name='3')
		mu = linear(tf.nn.elu(l1), [512,FLAGS.num_latents], name='mu')
		rho = linear(tf.nn.elu(l1), [512,FLAGS.num_latents], name='rho')
		#log_sigma = tf.nn.softplus(rho)
		log_sigma = rho
		return mu, log_sigma

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
		l1 = linear(z, [z.get_shape()[1], 512], name='5')
		#l2 = linear(tf.nn.elu(l1), [256,500], name='6')
		#l3 = linear(tf.nn.elu(l2), [500,500], name='7')
		#l3 = linear(tf.nn.elu(l3), [500,500], name='7b')
		#l3 = linear(tf.nn.elu(l3), [500,500], name='7c')
		return tf.nn.sigmoid(linear(tf.nn.elu(l1), [512,784], name='8'))
	
##################EVALUATION##########################
def debug_latents_mlp(opt, data, sample, img_transforms, feat_transforms, angle_transforms):
	#create the model
	#sample input
	x = tf.placeholder(tf.float32, [1,28,28,1], name='x')
	#random initial transformation
	#t_params_initial_tf = tf.placeholder(tf.float32, [1,6], name='t_params_initial')
	#...and apply to image
	#x_initial = tf.reshape(transformer(x, t_params_initial_tf, [28, 28]), [1,28, 28, 1])
	
	#transformation parameters
	t_params = tf.placeholder(tf.float32, [1,6], name='t_params')
	#input image under transformation
	x_transformed = transformer(x, t_params, [28,28])
	x_transformed.set_shape([1, 28, 28, 1])

	#the reference latents
	z_reference_tf = tf.placeholder(tf.float32, [1, FLAGS.num_latents], name='z_reference')

	#build the encoder
	with tf.variable_scope('mainModel', reuse=False):
		with tf.variable_scope("Encoder", reuse=False):
			z = encoder(flatten(x))
		with tf.variable_scope('Encoder', reuse=True):
			z_for_transformed = encoder(flatten(x_transformed))
	with tf.variable_scope('mainModel', reuse=False):
		with tf.variable_scope('Decoder', reuse=False):
			recon = decoder(z_for_transformed)

	if FLAGS.eq_dim > 0:
		
		z_for_transformed_eq, z_for_transformed_inv = tf.split(z_for_transformed,
			[FLAGS.eq_dim, z_for_transformed.get_shape().as_list()[1] - FLAGS.eq_dim], axis=1)

		z_reference_tf_eq, z_reference_tf_inv = tf.split(z_reference_tf,
			[FLAGS.eq_dim, z_reference_tf.get_shape().as_list()[1] - FLAGS.eq_dim], axis=1)
		
		l_shape = z_for_transformed_eq.get_shape()
		#angle in latent space
		inv_rep = tf.reshape(z_for_transformed_eq, tf.stack([l_shape[0],l_shape[1]/2,2]))
		inv_vec = tf.reduce_sum(tf.square(inv_rep),axis=2)

		inv_rep_ref = tf.reshape(z_reference_tf_eq, tf.stack([l_shape[0],l_shape[1]/2,2]))
		inv_vec_ref = tf.reduce_sum(tf.square(inv_rep_ref),axis=2)

		norm_inv_vec = tf.reduce_mean(tf.square(inv_vec), axis=1)

		angle_tf = tf.sqrt(tf.reduce_mean(tf.square(inv_vec_ref - inv_vec), axis=1) / (norm_inv_vec))
		per_invariant_diff = tf.square(inv_vec_ref - inv_vec) #/ (tf.square(inv_vec) + 1e-3)

	else:
		l_shape = z_for_transformed.get_shape()
		#angle in latent space
		inv_rep = tf.reshape(z_for_transformed, tf.stack([l_shape[0],l_shape[1]/2,2]))
		inv_vec = tf.reduce_sum(tf.square(inv_rep),axis=2)

		inv_rep_ref = tf.reshape(z_reference_tf, tf.stack([l_shape[0],l_shape[1]/2,2]))
		inv_vec_ref = tf.reduce_sum(tf.square(inv_rep_ref),axis=2)

		norm_inv_vec = tf.reduce_mean(tf.square(inv_vec), axis=1)

		angle_tf = tf.sqrt(tf.reduce_mean(tf.square(inv_vec_ref - inv_vec), axis=1) / (norm_inv_vec))
		per_invariant_diff = tf.square(inv_vec_ref - inv_vec) #/ (tf.square(inv_vec) + 1e-3)

	#model for forward reconstruction
	f_params = tf.placeholder(tf.float32, [1,2,2], name='f_params')
	if FLAGS.eq_dim == - 1:
		forward_z = el.feature_space_transform2d(z, [z.get_shape()[0], z.get_shape()[1]], f_params)
	else:
		eq_part, inv_part = tf.split(z, [FLAGS.eq_dim, z.get_shape().as_list()[1] - FLAGS.eq_dim], axis=1)
		eq_part = el.feature_space_transform2d(eq_part, [z.get_shape()[0], FLAGS.eq_dim], f_params)
		forward_z = tf.concat([eq_part, inv_part], 1)
		
	with tf.variable_scope('mainModel', reuse=False):
		with tf.variable_scope('Decoder', reuse=True):
			recon_forward = decoder(forward_z)

	#now start the session
	sess = tf.Session()

	#create tf stuff 
	print('Attempting to restore from [{:s}]'.format(opt['save_path']))
	saver = tf.train.Saver()
	#saver.restore(sess, tf.train.latest_checkpoint(opt['save_path']))
	saver.restore(sess, "/home/sgarbin/Projects/harmonicConvolutions/tensorflow1/scale/checkpoints/autotrain_bn/model.ckpt-190")
	#pick a random initial transformation
	t_params_initial, _ = el.random_transform(1, opt['im_size'])

	#get reference latents
	z_reference = sess.run(z, feed_dict={
		x : sample,
		#t_params_initial_tf: t_params_initial
	})
	print('Got initial latents, now performing tests...')
	checkFolder('samples/debug_imgs')
	removeAllFilesInDirectory('samples/debug_imgs', '.*')
	#get latents for all other inputs
	latent_angles = []
	per_invariant_diffs = []
	for img_t, feat_t, idx in zip(img_transforms, feat_transforms, xrange(len(img_transforms))):
		latent_angle, img_initial, img_recon, img_recon_foward, inv_diff = sess.run([angle_tf, x_transformed, recon, recon_forward, per_invariant_diff], feed_dict={
			x: sample,
			#t_params_initial_tf: t_params_initial,
			z_reference_tf: z_reference,
			t_params: img_t,
			f_params: feat_t
		})
		latent_angles.append(latent_angle[0])
		per_invariant_diffs.append(inv_diff)
		skio.imsave('samples/debug_imgs/img_input' + str(idx) + '.png', np.clip(np.reshape(img_initial, [28, 28]), 0, 1))
		skio.imsave('samples/debug_imgs/img_reconstruction' + str(idx) + '.png', np.clip(np.reshape(img_recon, [28, 28]), 0, 1))
		skio.imsave('samples/debug_imgs/img_forward' + str(idx) + '.png', np.clip(np.reshape(img_recon_foward, [28, 28]), 0, 1))
	print(angle_transforms)
	print(latent_angles)

	#differences_per_invariant = np.transpose(np.squeeze(np.asarray(per_invariant_diffs)))
	differences_per_invariant = np.squeeze(np.asarray(per_invariant_diffs))
	plt.plot(differences_per_invariant)
	plt.ylabel('invariant differences')
	plt.show()



######################################################

def debug(opt, data):
	sample = data['X']['valid'][np.newaxis,np.random.randint(5000),...]
	#generate image space and latent space transformations
	max_angles = 360
	image_transforms = []
	feature_transforms = []
	angle_transforms = []
	for i in xrange(max_angles):
		theta = 2.*np.pi*i/(1.*max_angles)
		img_t, feat_t = el.random_transform_theta(1, opt['im_size'],
			theta) # last arg is theta
		#print(img_t)
		#print(feat_t)
		image_transforms.append(img_t)
		feature_transforms.append(feat_t)
		angle_transforms.append(theta)

	debug_latents_mlp(opt, data, sample, image_transforms, feature_transforms, angle_transforms)

def train(inputs, outputs, ops, opt, data):
	"""Training loop"""
	x, global_step, t_params_initial, t_params, f_params, lr, xs, fs_params, ts_params, random_sample = inputs
	loss, merged, recon = outputs
	train_op = ops
	
	# For checkpoints
	saver = tf.train.Saver()
	gs = 0
	start = time.time()
	n_train = data['X']['train'].shape[0]
	n_valid = data['X']['valid'].shape[0]
	n_test = data['X']['test'].shape[0]
	sess = tf.Session()
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
		
		#save model
		if epoch % FLAGS.save_step == 0:
			path = saver.save(sess, opt['save_path'], epoch)
			print('Saved model to ' + path)
		# Validation
		if epoch % 10 == 0:
			Recon = []
			sample = data['X']['valid'][np.newaxis,np.random.randint(5000),...]
			#noise_sample = np.random.normal(size=[1, FLAGS.num_latents])
			noise_sample = np.ones([1, FLAGS.num_latents])
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
							t_params_initial: tp_init,
							random_sample: noise_sample}
				Recon.append(sess.run(ops, feed_dict=feed_dict))
			
			samples_ = np.reshape(Recon, (-1,28,28))
			
			ns = 20
			sh = 28
			tile_image = np.zeros((ns*sh,ns*sh))
			for j in xrange(ns*ns):
				m = sh*(j/ns) 
				n = sh*(j%ns)
				tile_image[m:m+sh,n:n+sh] = 1.-samples_[j,...]
			save_name = './samples/vae/image_%04d.png' % epoch
			skio.imsave(save_name, tile_image)

def main(_):
	opt = {}
	"""Main loop"""
	tf.reset_default_graph()
	opt['root'] = '/home/sgarbin'
	dir_ = opt['root'] + '/Projects/harmonicConvolutions/tensorflow1/scale'
	opt['mb_size'] = 128
	opt['n_channels'] = 10
	opt['n_epochs'] = 2000
	opt['lr_schedule'] = [50, 75]
	opt['lr'] = 1e-3
	opt['im_size'] = (28,28)
	opt['train_size'] = 55000
	opt['equivariant_weight'] = 1 #1e-3
	flag = 'vae'
	opt['convolutional'] = False
	if opt['convolutional']:
		opt['summary_path'] = dir_ + '/summaries/autotrain_conv_{:s}'.format(flag)
		opt['save_path'] = dir_ + '/checkpoints/autotrain_conv_{:s}/model.ckpt'.format(flag)
	else:
		opt['summary_path'] = dir_ + '/summaries/autotrain_{:s}'.format(flag)
		opt['save_path'] = dir_ + '/checkpoints/autotrain_{:s}/model.ckpt'.format(flag)
	opt['loss_type_image'] = 'l2'

	#check and clear directories
	checkFolder(opt['summary_path'])
	checkFolder(opt['save_path'])
	removeAllFilesInDirectory(opt['summary_path'], '.*')
	removeAllFilesInDirectory(opt['save_path'], '.*')


	data = load_data()
	data['X']['train'] = data['X']['train'][:opt['train_size'],:]
	data['Y']['train'] = data['Y']['train'][:opt['train_size'],:]

	if FLAGS.ANALISE:
		debug(opt, data)
		sys.exit(0)

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
	ts_params = tf.placeholder(tf.float32, [1,6], name='ts_params') #image
	fs_params = tf.placeholder(tf.float32, [1,2,2], name='fs_params') #latents
	lr = tf.placeholder(tf.float32, [], name='lr')
	
	random_sample = tf.placeholder(tf.float32, [1,FLAGS.num_latents], name='random_sample')
	
	# Build the model
	#transform input
	shape_temp = x.get_shape()
	x_initial_transform = transformer(x, t_params_initial, (28,28))
	x_initial_transform.set_shape(shape_temp)
	#build encoder
	reconstruction, latents, mu, log_sigma, eps = single_model_non_siamese(x_initial_transform,
		f_params, t_params=t_params, conv=opt['convolutional'])
	reconstruction = tf.reshape(reconstruction, x.get_shape())
	#transform input corresponding to latents for loss
	reconstruction_transform = transformer(x_initial_transform, t_params, (28,28))

	kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + 2.0 * log_sigma - tf.square(mu) - tf.exp(2.0 * log_sigma), axis=1))
	img_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reconstruction_transform - reconstruction), axis=(1,2)))
	loss = img_loss + kl_loss

	recon = single_model(xs, fs_params, name_scope='mainModel', conv=opt['convolutional'],
		t_params=ts_params, random_sample=random_sample)

	tf.summary.scalar('Loss', loss)
	tf.summary.scalar('Loss_KL', kl_loss)
	tf.summary.scalar('Loss_Img', img_loss)
	tf.summary.scalar('LearningRate', lr)
	
	merged = tf.summary.merge_all()
	
	# Build optimizer
	optim = tf.train.AdamOptimizer(lr)
	train_op = optim.minimize(loss, global_step=global_step)
	
	inputs = [x, global_step, t_params_initial, t_params, f_params, lr, xs, fs_params, ts_params, random_sample]
	outputs = [loss, merged, recon]
	ops = [train_op]
	
	# Train
	return train(inputs, outputs, ops, opt, data)

if __name__ == '__main__':
	tf.app.run()
