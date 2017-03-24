'''Autoencoder'''

import os
import sys
import time
import glob
sys.path.append('../')

#import cv2
#import input_data
import numpy as np
import skimage.color as skco
import skimage.io as skio
import skimage.transform as sktr
import tensorflow as tf

import equivariant_loss as el
#import face_loader
import models

from spatial_transformer import transformer
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


################ DATA #################

#-----------ARGS----------
flags = tf.app.flags
FLAGS = flags.FLAGS
#execution modes
flags.DEFINE_boolean('ANALYSE', False, 'runs model analysis')
flags.DEFINE_integer('eq_dim', -1, 'number of latent units to rotate')
flags.DEFINE_integer('num_latents', 30, 'Dimension of the latent variables')
flags.DEFINE_float('l2_latent_reg', 1e-6, 'Strength of l2 regularisation on latents')
flags.DEFINE_integer('save_step', 500, 'Interval (epoch) for which to save')
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
############## MODEL ####################
def leaky_relu(x, leak=0.1):
	"""The leaky ReLU nonlinearity"""
	return tf.nn.relu(x) + tf.nn.relu(-leak*x)


def transformer_layer(x, t_params, imsh):
	"""Spatial transformer wtih shapes sets"""
	xsh = x.get_shape()
	x_in = transformer(x, t_params, imsh)
	x_in.set_shape(xsh)
	return x_in


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
	return tf.reshape(input, [s[0], num_params])


def linear(x, shape, name='0', bias_init=0.01):
	"""Basic linear matmul layer"""
	with tf.variable_scope('linear') as scope:
		He_initializer = tf.contrib.layers.variance_scaling_initializer()
		W = tf.get_variable(name+'_W', shape=shape, initializer=He_initializer)
		z = tf.matmul(x, W, name='mul'+str(name))
		return bias_add(z, shape[1], bias_init=bias_init, name=name)


def bias_add(x, nc, bias_init=0.01, name='0'):
	with tf.variable_scope('bias') as scope:
		const_initializer = tf.constant_initializer(value=bias_init)
		b = tf.get_variable(name+'_b', shape=nc, initializer=const_initializer)
		return tf.nn.bias_add(x, b)


def autoencoder(x, f_params, is_training, opt, reuse=False):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	with tf.variable_scope('mainModel', reuse=reuse) as scope:
		with tf.variable_scope("encoder", reuse=reuse) as scope:
			z = encoder(x, is_training, opt, reuse=reuse)
		with tf.variable_scope("feature_transformer", reuse=reuse) as scope:
			matrix_shape = [xsh[0], z.get_shape()[1]]
			z_ = el.feature_transform_matrix_n(z, matrix_shape, f_params)
		with tf.variable_scope("decoder", reuse=reuse) as scope:
			r = decoder(z_, is_training, opt, reuse=reuse)
	return r, z


def autoencoder_efros(x, d_params, is_training, opt, reuse=False):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	with tf.variable_scope('mainModel', reuse=reuse) as scope:
		with tf.variable_scope("encoder", reuse=reuse) as scope:
			z = encoder(x, is_training, opt, reuse=reuse)
		with tf.variable_scope("efros_transformer", reuse=reuse) as scope:
			et1 = linear(d_params, [4, 128], name='et1')
			et2 = linear(leaky_relu(et1), [128, 256], name='et2')
			z = tf.concat([et2,z], 1)
		with tf.variable_scope("decoder", reuse=reuse) as scope:
			r = decoder(z, is_training, opt, reuse=reuse, n_in=1026+256)
	return r


def encoder(x, is_training, opt, reuse=False):
	"""Encoder MLP"""
	nl = opt['nonlinearity']
	
	with tf.variable_scope('encoder_1') as scope:
		l1 = conv(x, [5,5,opt['color'],32], stride=1, name='e1', padding='VALID')
		l1 = bn4d(l1, is_training, reuse=reuse, name='bn1')
	
	with tf.variable_scope('encoder_2') as scope:
		l2 = conv(nl(l1), [3,3,32,64], stride=2, name='e2', padding='SAME')
		l2 = bn4d(l2, is_training, reuse=reuse, name='bn2')
	
	with tf.variable_scope('encoder_3') as scope:
		l3 = conv(nl(l2), [3,3,64,128], stride=2, name='e3', padding='SAME')
		l3 = bn4d(l3, is_training, reuse=reuse, name='bn3')
	
	with tf.variable_scope('encoder_4') as scope:
		l4 = conv(nl(l3), [3,3,128,256], stride=2, name='e4', padding='SAME')
		l4 = bn4d(l4, is_training, reuse=reuse, name='bn4')
	
	with tf.variable_scope('encoder_5') as scope:
		l5 = conv(nl(l4), [3,3,256,512], stride=2, name='e5', padding='SAME')
		l5 = bn4d(l5, is_training, reuse=reuse, name='bn5')
	
	with tf.variable_scope('encoder_6') as scope:
		l6 = conv(nl(l5), [3,3,512,1024], stride=2, name='e6', padding='SAME')
		l6 = bn4d(l6, is_training, reuse=reuse, name='bn6')
		l6 = tf.reduce_mean(l6, axis=(1,2))
	
	with tf.variable_scope('encoder_mid') as scope:
		return linear(nl(l6), [1024,1026], name='e_out')


def decoder(z, is_training, opt, reuse=False, n_in=1026):
	"""Encoder MLP"""
	nl = opt['nonlinearity']
	
	with tf.variable_scope('decoder_mid') as scope:
		l_in = linear(z, [n_in,2*2*1024], name='d_in')
		l_in = tf.reshape(l_in, shape=(-1,2,2,1024))
	
	with tf.variable_scope('decoder_6') as scope:
		l6 = deconv(nl(l_in), [4,4,1024,512], [5,5], name='d6')
		l6 = bn4d(l6, is_training, reuse=reuse, name='bn6')
	
	with tf.variable_scope('decoder_5') as scope:
		l5 = deconv(nl(l6), [5,5,512,256], [9,9], name='d5')
		l5 = bn4d(l5, is_training, reuse=reuse, name='bn5')
	
	with tf.variable_scope('decoder_4') as scope:
		l4 = deconv(nl(l5), [5,5,256,128], [18,18], name='d4')
		l4 = bn4d(l4, is_training, reuse=reuse, name='bn4')
	
	with tf.variable_scope('decoder_3') as scope:
		l3 = deconv(nl(l4), [5,5,128,64], [37,37], name='d3')
		l3 = bn4d(l3, is_training, reuse=reuse, name='bn3')
	
	with tf.variable_scope('decoder_2') as scope:
		l2 = deconv(nl(l3), [5,5,64,32], [75,75], name='d2')
		l2 = bn4d(l2, is_training, reuse=reuse, name='bn2')
	
	with tf.variable_scope('decoder_1') as scope:
		return deconv(nl(l2), [5,5,32,opt['color']], opt['im_size'], name='d1')

'''
def encoder(x, is_training, opt, reuse=False):
	"""Encoder MLP"""
	nl = opt['nonlinearity']
	
	with tf.variable_scope('encoder_1') as scope:
		l1 = conv(x, [5,5,opt['color'],16], stride=1, name='e1', padding='VALID')
		l1 = bn4d(l1, is_training, reuse=reuse, name='bn1')
	
	with tf.variable_scope('encoder_2') as scope:
		l2 = conv(nl(l1), [3,3,16,32], stride=2, name='e2', padding='SAME')
		l2 = bn4d(l2, is_training, reuse=reuse, name='bn2')
	
	with tf.variable_scope('encoder_3') as scope:
		l3 = conv(nl(l2), [3,3,32,64], stride=2, name='e3', padding='SAME')
		l3 = bn4d(l3, is_training, reuse=reuse, name='bn3')
	
	with tf.variable_scope('encoder_4') as scope:
		l4 = conv(nl(l3), [3,3,64,128], stride=2, name='e4', padding='SAME')
		l4 = bn4d(l4, is_training, reuse=reuse, name='bn4')
	
	with tf.variable_scope('encoder_5') as scope:
		l5 = conv(nl(l4), [3,3,128,256], stride=2, name='e5', padding='SAME')
		l5 = bn4d(l5, is_training, reuse=reuse, name='bn5')
	
	with tf.variable_scope('encoder_6') as scope:
		l6 = conv(nl(l5), [3,3,256,512], stride=2, name='e6', padding='SAME')
		l6 = bn4d(l6, is_training, reuse=reuse, name='bn6')
		l6 = tf.reduce_mean(l6, axis=(1,2))
	
	with tf.variable_scope('encoder_mid') as scope:
		return linear(nl(l6), [512,510], name='e_out')


def decoder(z, is_training, opt, reuse=False, n_in=510):
	"""Encoder MLP"""
	nl = opt['nonlinearity']
	
	with tf.variable_scope('decoder_mid') as scope:
		l_in = linear(z, [n_in,2*2*512], name='d_in')
		l_in = tf.reshape(l_in, shape=(-1,2,2,512))
	
	with tf.variable_scope('decoder_6') as scope:
		l6 = deconv(nl(l_in), [4,4,1024,256], [5,5], name='d6')
		l6 = bn4d(l6, is_training, reuse=reuse, name='bn6')
	
	with tf.variable_scope('decoder_5') as scope:
		l5 = deconv(nl(l6), [5,5,256,128], [9,9], name='d5')
		l5 = bn4d(l5, is_training, reuse=reuse, name='bn5')
	
	with tf.variable_scope('decoder_4') as scope:
		l4 = deconv(nl(l5), [5,5,128,64], [18,18], name='d4')
		l4 = bn4d(l4, is_training, reuse=reuse, name='bn4')
	
	with tf.variable_scope('decoder_3') as scope:
		l3 = deconv(nl(l4), [5,5,64,32], [37,37], name='d3')
		l3 = bn4d(l3, is_training, reuse=reuse, name='bn3')
	
	with tf.variable_scope('decoder_2') as scope:
		l2 = deconv(nl(l3), [5,5,32,16], [75,75], name='d2')
		l2 = bn4d(l2, is_training, reuse=reuse, name='bn2')
	
	with tf.variable_scope('decoder_1') as scope:
		return deconv(nl(l2), [5,5,16,opt['color']], opt['im_size'], name='d1')
'''

############################## DC-IGN ##########################################
def autoencoder_DCIGN(x, f_params, is_training, opt, reuse=False):
	"""Build a model to rotate features"""
	xsh = x.get_shape().as_list()
	with tf.variable_scope('mainModel', reuse=reuse) as scope:
		with tf.variable_scope("encoder", reuse=reuse) as scope:
			z = encoder_DCIGN(x, is_training, opt, reuse=reuse)
		with tf.variable_scope("feature_transformer", reuse=reuse) as scope:
			matrix_shape = [xsh[0], z.get_shape()[1]]
			z_ = el.feature_transform_matrix_n(z, matrix_shape, f_params)
		with tf.variable_scope("decoder", reuse=reuse) as scope:
			r = decoder_DCIGN(z_, is_training, opt, reuse=reuse)
	return r, z


def encoder_DCIGN(x, is_training, opt, reuse=False):
	"""Encoder MLP"""
	nl = tf.nn.relu #opt['nonlinearity']
	
	with tf.variable_scope('encoder_1') as scope:
		l1 = conv(x, [5,5,opt['color'],96], name='e1', padding='VALID')
		l1 = tf.nn.max_pool(l1, (1,2,2,1), (1,2,2,1), 'VALID')
		l1 = bn4d(l1, is_training, reuse=reuse, name='bn1')
	
	with tf.variable_scope('encoder_2') as scope:
		l2 = conv(nl(l1), [5,5,96,64], name='e2', padding='VALID')
		l2 = tf.nn.max_pool(l2, (1,2,2,1), (1,2,2,1), 'VALID')
		l2 = bn4d(l2, is_training, reuse=reuse, name='bn2')
	
	with tf.variable_scope('encoder_3') as scope:
		l3 = conv(nl(l2), [5,5,64,32], name='e3', padding='VALID')
		l3 = tf.nn.max_pool(l3, (1,2,2,1), (1,2,2,1), 'VALID')
		l3 = tf.reshape(l3, shape=(-1,32*15*15))

	with tf.variable_scope('encoder_mid') as scope:
		return linear(nl(l3), [32*15*15,204], name='e_out')


def decoder_DCIGN(z, is_training, opt, reuse=False):
	"""Encoder MLP"""
	nl = tf.nn.relu #opt['nonlinearity']
	
	with tf.variable_scope('decoder_mid') as scope:
		l_in = linear(z, [204,32*15*15], name='d_in')
		l_in = tf.reshape(l_in, shape=(-1,15,15,32))
	
	with tf.variable_scope('decoder_4') as scope:
		l4 = deconv(nl(l_in), [7,7,32,64], [30,30], name='d4', padding='VALID')
		l4 = bn4d(l4, is_training, reuse=reuse, name='bn4')
	
	with tf.variable_scope('decoder_3') as scope:
		l3 = deconv(nl(l4), [7,7,64,96], [48,48], name='d3', padding='VALID')
		l3 = bn4d(l3, is_training, reuse=reuse, name='bn3')
	
	with tf.variable_scope('decoder_2') as scope:
		l2 = deconv(nl(l3), [7,7,96,96], [84,84], name='d2', padding='VALID')
		l2 = bn4d(l2, is_training, reuse=reuse, name='bn2')
	
	with tf.variable_scope('decoder_1') as scope:
		return deconv(nl(l2), [7,7,96,opt['color']], [156,156], name='d1', padding='VALID')


###############################################################################


def bernoulli_xentropy(target, recon, mean=False):
	"""Cross-entropy for Bernoulli variables"""
	x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=recon)
	return mean_loss(x_entropy, mean=mean)


def gaussian_nll(target, recon, mean=False):
	"""L2 loss"""
	loss = tf.square(target - recon)
	return mean_loss(loss, mean=mean)


def SSIM(target, recon, mean=False):
	C1 = 0.01 ** 2
	C2 = 0.03 ** 2
	
	mu_x = tf.nn.avg_pool(target, (1,3,3,1), (1,1,1,1), 'VALID')
	mu_y = tf.nn.avg_pool(recon, (1,3,3,1), (1,1,1,1), 'VALID')
	
	sigma_x  = tf.nn.avg_pool(target ** 2, (1,3,3,1), (1,1,1,1), 'VALID') - mu_x ** 2
	sigma_y  = tf.nn.avg_pool(recon ** 2, (1,3,3,1), (1,1,1,1), 'VALID') - mu_y ** 2
	sigma_xy = tf.nn.avg_pool(target * recon , (1,3,3,1), (1,1,1,1), 'VALID') - mu_x * mu_y
	
	SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
	SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
	
	SSIM = SSIM_n / SSIM_d
	
	loss = tf.clip_by_value((1 - SSIM) / 2, 0, 1)
	return mean_loss(loss, mean=mean)


def mean_loss(loss, mean=False):
	if mean:
		loss = tf.reduce_mean(loss)
	else:
		loss = tf.reduce_mean(tf.reduce_sum(loss))
	return loss


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


############################################
def validate(inputs, outputs, opt):
	"""Validate network"""
	# Unpack inputs, outputs and ops
	x, geometry, lighting, is_training = inputs
	recon = outputs[0]
	
	save_dir = '{:s}/Code/harmonicConvolutions/tensorflow1/scale'.format(opt['root'])
	
	save_folder = '{:s}/movie_faces/supp_mat/'.format(save_dir)
	#DOF = ['az_light','az_rot','el_light','el_rot']
	DOF = ['joint']
	faces = ['face010','face009','face008','face007','face006']
	saver = tf.train.Saver()
	with tf.Session() as sess:
		# Load variables from stored model
		saver.restore(sess, opt['save_path'])
		for j, dof in enumerate(DOF):
			for l, face in enumerate(faces):
				X = skio.imread('{:s}/az_rot/{:s}/-01_000_014_014.png'.format(opt['data_folder'],face))[np.newaxis,...]
				#X = skio.imread('/home/dworrall/Data/obama.png'.format(opt['data_folder']))
				#X = X[np.newaxis,...,:3]
				X = X.astype(np.float32)
				#canvas = []
				#canvas.append(sktr.resize(np.squeeze(X)/255., (300,300), order=0))
		
				# Angular limits
				#lo = np.asarray([-114,-86,-114,-30]) * (np.pi/180.)
				#hi = np.asarray([114,86,114,30]) * (np.pi/180.)
				
				# Create folder
				path = '{:s}{:s}/{:s}'.format(save_folder, dof, face)
				if not os.path.exists(path):
					os.mkdir(path)
				# loop
				#for i, var in enumerate(np.linspace(lo[j], hi[j], num=300)):
				#for i, var in enumerate(np.linspace(0,135*(np.pi/180.), num=6)):
				for i, var in enumerate(np.linspace(0.,2*np.pi, num=300)):
					# Defaults
					#phi = 0. #-29 * (np.pi/180.)
					#theta = 0. * (np.pi/180.)
					phi_light = 14 * (np.pi/180.)
					theta_light = 14 * (np.pi/180.)
					phi = 50*np.cos(var) * (np.pi/180.)
					theta = 20*np.sin(var) * (np.pi/180.)
					# Assign var
					#if j == 0:
					#	phi_light = var
					#elif j == 1:
					#	phi = -var
					#elif j == 2:
					#	theta_light = var
					#elif j == 3:
					#	theta = var
					Geometry = rot3d(phi, theta)[np.newaxis,...].astype(np.float32)
					Lighting = rot3d(phi_light, theta_light)[np.newaxis,...].astype(np.float32)
					
					feed_dict = {x: X,
									 geometry: Geometry,
									 lighting: Lighting,
									 is_training: False}
					Recon = sess.run(recon, feed_dict=feed_dict)
					Recon_ = sktr.resize(Recon[0,...], (300,300), order=0)
					skio.imsave('{:s}{:s}/{:s}/{:04d}.png'.format(save_folder, dof, face, i), Recon_)
					#canvas.append(np.hstack((np.zeros((300,5,3)),Recon_)))
	#canvas = np.hstack(canvas)
	#canvas = skco.rgb2gray(canvas)
	#skio.imsave('{:s}/canvas.png'.format(save_folder), canvas)
	#skio.imsave('/home/dworrall/Data/real_faces_light.png', canvas)


def trajectory(inputs, outputs, opt):
	"""Validate network"""
	# Unpack inputs, outputs and ops
	x, geometry, lighting, is_training = inputs
	recon = outputs[0]
	
	save_dir = '{:s}/Code/harmonicConvolutions/tensorflow1/scale'.format(opt['root'])
	
	save_folder = '{:s}/movie_faces/supp_mat/'.format(save_dir)
	faces = ['face004']
	saver = tf.train.Saver()
	with tf.Session() as sess:
		# Load variables from stored model
		saver.restore(sess, opt['save_path'])
		print('Hello')
		for l, face in enumerate(faces):
			X = skio.imread('{:s}/az_rot/{:s}/000_000_014_014.png'.format(opt['data_folder'],face))[np.newaxis,...]
			X = X.astype(np.float32)
			
			# Create folder
			path = '{:s}{:s}/{:s}'.format(save_folder, dof, face)
			if not os.path.exists(path):
				os.mkdir(path)
			for i, var in enumerate(np.linspace(0.,2*np.pi, num=300)):
				# Defaults
				phi_light = 14 * (np.pi/180.)
				theta_light = 14 * (np.pi/180.)
				phi = 50*np.cos(var) * (np.pi/180.)
				theta = 20*np.sin(var) * (np.pi/180.)
				Geometry = rot3d(phi, theta)[np.newaxis,...].astype(np.float32)
				Lighting = rot3d(phi_light, theta_light)[np.newaxis,...].astype(np.float32)
				
				feed_dict = {x: X,
								 geometry: Geometry,
								 lighting: Lighting,
								 is_training: False}
				Recon = sess.run(recon, feed_dict=feed_dict)
				Recon_ = sktr.resize(Recon[0,...], (300,300), order=0)
				print i
				skio.imsave('{:s}traj/{:s}/{:04d}.png'.format(save_folder, face, i), Recon_)


def feature_stability(inputs, outputs, opt):
	"""Training loop"""
	# Unpack inputs, outputs and ops
	x, geometry, lighting, is_training = inputs
	recon, latents = outputs
	
	dataset = ['az_rot', 'el_rot', 'az_light', 'el_light']
	labels = ['Azimuth (degrees)','Elevation (degrees)','Lighting azimuth (degrees)','Lighting elevation (degrees)']
	limit = [43,15,57,57]
	
	template = skio.imread('{:s}/az_rot/face001/041_000_014_014.png'.format(opt['data_folder']))[np.newaxis,...]
	#template = skio.imread('{:s}/el_light/face000/000_000_000_001.png'.format(opt['data_folder']))[np.newaxis,...]
	Geometry = np.eye(3)[np.newaxis,...].astype(np.float32)
	Lighting = np.eye(3)[np.newaxis,...].astype(np.float32)
	feed_dict = {x: template, geometry: Geometry, lighting: Lighting, is_training: False}
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		# Load variables from stored model
		saver.restore(sess, opt['save_path'])
		Template = get_invariant(sess.run(latents, feed_dict=feed_dict))
		for d in xrange(4):
			colorB=iter(cm.Blues(np.linspace(0.3,0.7,10)))
			colorR=iter(cm.Reds(np.linspace(0.3,0.7,10)))
			for j in xrange(10):
				angle = []
				Latents_list = []
				X, fnames = load_images('{:s}/{:s}/face{:03d}/'.format(opt['data_folder'],dataset[d],j), d)
				# Test -- variables beginning with a capital letter are not part of the
				# TF Graph
				for i, fname in enumerate(fnames):
					#Geometry = rot3d(phi, theta)[np.newaxis,...].astype(np.float32)
					#Lighting = rot3d(phi_light, theta_light)[np.newaxis,...].astype(np.float32)
					Geometry = np.eye(3)[np.newaxis,...].astype(np.float32)
					Lighting = np.eye(3)[np.newaxis,...].astype(np.float32)
					if opt['color'] == 1:
						X_ = skco.rgb2gray(X[fname])[...,np.newaxis]
					else:
						X_ = X[fname]
		
					ops = [recon, latents]
					feed_dict = {x: X_,
									 geometry: Geometry,
									 lighting: Lighting,
									 is_training: False}
					Recon, Latents = sess.run(ops, feed_dict=feed_dict)
					Recon_ = sktr.resize(Recon[0,...], (300,300), order=0)
					if opt['color'] == 1:
						Recon_ = Recon_[:,:,0]
					#skio.imsave('{:s}/{:04d}.png'.format(opt['movie_faces'], i), Recon_)
					Latents_list.append(Latents)
					angle.append(fname)
				
				errors = []
				Terrors = []
				L_canon = get_invariant(Latents_list[len(Latents_list)/2])
				for L in Latents_list:
					L = get_invariant(L)
					error = np.dot(L,L_canon) / np.sqrt(np.sum(L_canon**2)*np.sum(L**2))
					terror = np.dot(L,Template) / np.sqrt(np.sum(Template**2)*np.sum(L**2))
					#error = np.sqrt(np.mean((L - L_canon)**2))
					#terror = np.sqrt(np.mean((L - Template)**2))
					errors.append(error)
					Terrors.append(terror)
				plt.plot(angle, errors, c=next(colorB))
				plt.plot(angle, Terrors, c=next(colorR))
				
			# Plot training limits
			plt.plot((-limit[d],-limit[d]),(0.8,1.002),'k--')
			plt.plot((limit[d],limit[d]),(0.8,1.002),'k--')
				
			plt.tick_params(axis='both', which='major', labelsize=30)
			plt.xlim([np.amin(angle),np.amax(angle)])
			plt.ylim([0.8,1.002])
			plt.xlabel(labels[d], fontsize=30)
			if d > 0:
				plt.yticks([], [])
			else:
				plt.ylabel('Cosine similarity', fontsize=16)
				#plt.ylabel('L2 distance', fontsize=30)
			plt.tight_layout()
			plt.savefig('{:s}/mixed_face/mf_{:s}_.pdf'.format(opt['movie_faces'], dataset[d]))
			plt.clf()


def L2_distance(opt):
	"""Training loop"""
	dataset = ['az_rot', 'el_rot', 'az_light', 'el_light']
	labels = ['Azimuth (degrees)','Elevation (degrees)','Lighting azimuth (degrees)','Lighting elevation (degrees)']
	
	template = skio.imread('{:s}/az_rot/face001/041_000_014_014.png'.format(opt['data_folder']))[np.newaxis,...]
	for d in xrange(4):
		colorB=iter(cm.Blues(np.linspace(0.3,0.7,10)))
		colorR=iter(cm.Reds(np.linspace(0.3,0.7,10)))
		for j in xrange(10):
			angle = []
			errors = []
			Terrors = []
			X, fnames = load_images('{:s}/{:s}/face{:03d}/'.format(opt['data_folder'],dataset[d],j), d)
			key = np.sort(X.keys())[len(fnames)/2]
			X_canon = X[key]
			for i, fname in enumerate(fnames):
				X_ = X[fname]
				angle.append(fname)

				error = np.mean((X_ - X_canon)**2)
				terror = np.mean((X_ - template)**2)
				errors.append(error)
				Terrors.append(terror)
					
			plt.plot(angle, errors, c=next(colorB))
			plt.plot(angle, Terrors, c=next(colorR))
			
		plt.tick_params(axis='both', which='major', labelsize=16)
		plt.xlim([np.amin(angle),np.amax(angle)])
		#plt.ylim([0.8,1.002])
		plt.xlabel(labels[d], fontsize=16)
		plt.ylabel('L2 distance', fontsize=16)
		plt.tight_layout()
		plt.savefig('{:s}/mixed_face/mf_L2_{:s}.pdf'.format(opt['movie_faces'], dataset[d]))
		plt.clf()


def get_invariant(x):
	'''Implied batch size 1'''
	n = np.prod(x.shape)
	x1 = np.reshape(x, (n/6,1,6))
	x2 = np.reshape(x, (1,n/6,6))
	return np.reshape(np.sum(x2*x1, axis=2), -1)
	#return np.sum(np.reshape(x**2, (n/6,6)), axis=1)


def load_images(folder, index):
	"""Load images from folder, order, and return"""
	images = {}
	fnames = []
	for root, dirs, files in os.walk(folder):
		for f in files:
			fnames.append(int(f.replace('.png','').split('_')[index]))
			images[fnames[-1]] = skio.imread('{:s}/{:s}'.format(root,f))[np.newaxis,...]
	fnames = np.sort(fnames)
	return images, fnames


def rot3d(phi, theta):
	"""Compute the 3D rotation matrix for a roll-less transformation"""
	rotY = [[np.cos(phi),0.,-np.sin(phi)],
				[0.,1.,0.],
				[np.sin(phi),0.,np.cos(phi)]]
	rotZ = [[np.cos(theta),np.sin(theta),0.],
				[-np.sin(theta),np.cos(theta),0.],
				[0.,0.,1]]
	return np.dot(rotZ, rotY)


def get_latest_model(model_file):
	"""Model file"""
	dirname = os.path.dirname(model_file)
	basename = os.path.basename(model_file)
	nums = []
	print model_file
	for root, dirs, files in os.walk(dirname):
		for f in files:
			f = f.split('-')
			if f[0] == basename:
				nums.append(int(f[1].split('.')[0]))
	model_file += '-{:05d}'.format(max(nums))
	return model_file


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
		dir_ = '{:s}/Code/harmonicConvolutions/tensorflow1/scale'.format(opt['root'])
		opt['data_folder'] = '{:s}/Data/test_faces'.format(opt['root'])
	else:
		opt['root'] = '/home/sgarbin'
		dir_ = opt['root'] + '/Projects/harmonicConvolutions/tensorflow1/scale'
	opt['mb_size'] = 1
	opt['im_size'] = (150,150)
	opt['valid_size'] = 240
	opt['loss_type'] = 'SSIM_L1'
	opt['loss_weights'] = (0.85,0.15)
	opt['nonlinearity'] = leaky_relu
	opt['color'] = 3
	opt['method'] = 'worrall'
	if opt['method'] == 'kulkarni':
		opt['color'] = 1
	if opt['loss_type'] == 'SSIM_L1':
		lw = '_' + '_'.join([str(l) for l in opt['loss_weights']])
	else:
		lw = ''
	opt['movie_faces'] = '{:s}/movie_faces'.format(dir_)
	save_path = '{:s}/checkpoints/val_face15train_{:s}_{:s}{:s}/model.ckpt'.format(dir_, opt['loss_type'], opt['method'], lw)
	opt['save_path'] = get_latest_model(save_path)
	
	# Load data
	#valid_files = face_loader.get_files(opt['data_folder'])
	#x, target, geometry, lighting, d_params = face_loader.get_batches(valid_files, True, opt)
	#x /= 255.
	#target /= 255.

	# Placeholders
	x = tf.placeholder(tf.float32, [opt['mb_size'],opt['im_size'][0],opt['im_size'][1],opt['color']], name='x')
	geometry = tf.placeholder(tf.float32, [opt['mb_size'],3,3], name='f_params')
	lighting = tf.placeholder(tf.float32, [opt['mb_size'],3,3], name='f_params')
	is_training = tf.placeholder(tf.bool, [], name='is_training')
	
	# Build the training model
	zeros = tf.zeros_like(geometry)
	f_params1 = tf.concat([geometry,zeros], 1)
	f_params2 = tf.concat([zeros,lighting], 1)
	f_params = tf.concat([f_params1, f_params2], 2)
	
	x_ = x / 255.
	if opt['method'] == 'efros':
		recon, latents = autoencoder_efros(x_, d_params, is_training, opt)	
	elif opt['method'] == 'worrall':
		recon, latents = autoencoder(x_, f_params, is_training, opt)
	elif opt['method'] == 'kulkarni':
		recon, latents = autoencoder_DCIGN(x_, f_params, is_training, opt)
	recon = tf.nn.sigmoid(recon)


	# Set inputs, outputs, and training ops
	inputs = [x, geometry, lighting, is_training]
	outputs = [recon, latents]
	
	# Train
	#validate(inputs, outputs, opt)
	trajectory(inputs, outputs, opt)
	#feature_stability(inputs, outputs, opt)
	#L2_distance(opt)

if __name__ == '__main__':
	tf.app.run()


























