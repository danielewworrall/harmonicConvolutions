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
import face_loader
import models

from spatial_transformer import transformer


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
		l6 = deconv(nl(l_in), [4,4,512,256], [5,5], name='d6')
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
			z = el.feature_transform_matrix_n(z, matrix_shape, f_params)
		with tf.variable_scope("decoder", reuse=reuse) as scope:
			r = decoder_DCIGN(z, is_training, opt, reuse=reuse)
	return r


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


def classifier(x, opt):
	"""Linear classifier"""
	# Create invariant representation
	#x = tf.reshape(x, [opt['mb_size'],1026/6,6])
	#x3 = tf.reduce_sum(tf.square(x), axis=2)
	# Matrix of invariants
	x1 = tf.reshape(x, [opt['mb_size'],1026/6,1,6])
	x2 = tf.reshape(x, [opt['mb_size'],1,1026/6,6])
	#x2 = tf.reshape(x, [opt['mb_size'],1026/6,6])
	x3 = tf.reduce_mean(x1*x2, axis=3)
	x3 = tf.reshape(x3, (opt['mb_size'],(1026/6)**2))
	
	return linear(x3, [(1026/6)**2,1000])


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
def train(inputs, outputs, ops, summaries, opt):
	"""Training loop"""
	# Unpack inputs, outputs and ops
	lr, is_training = inputs
	loss = outputs[0]
	merged, recon_summary, target_summary, input_summary = summaries
	train_op, global_step = ops
	
	# For checkpoints
	gs = 0
	start = time.time()
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		# Initialize variables
		init = tf.global_variables_initializer()
		sess.run(init)
		# Threading and queueing
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		train_writer = tf.summary.FileWriter(opt['summary_path'], sess.graph)
		# Training loop
		try:
			while not coord.should_stop():
				# Learning rate
				exponent = sum([gs > i for i in opt['lr_schedule']])
				current_lr = opt['lr']*np.power(0.1, exponent)
				
				# Train
				ops = [global_step, loss, merged, train_op]
				feed_dict = {lr: current_lr, is_training: True, }
				gs, l, summary, __ = sess.run(ops, feed_dict=feed_dict)
				# Summary writers
				train_writer.add_summary(summary, gs)
				print('[{:06d} s | {:03d}]: {:01f}'.format(int(time.time()-start), gs, l))
				
				if gs % 250 == 0:
					ops = [recon_summary, target_summary, input_summary]
					rs, ts, is_ = sess.run(ops, feed_dict={is_training: False})
					train_writer.add_summary(rs, gs)
					train_writer.add_summary(ts, gs)
					train_writer.add_summary(is_, gs)
				
				# Save model
				if gs % FLAGS.save_step == 0:
					path = saver.save(sess, opt['save_path'], gs)
					print('Saved model to ' + path)
		except tf.errors.OutOfRangeError:
			pass
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
			coord.join(threads)
			

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
		opt['data_folder'] = '{:s}/Data/faces15'.format(opt['root'])
	else:
		opt['root'] = '/home/sgarbin'
		dir_ = opt['root'] + '/Projects/harmonicConvolutions/tensorflow1/scale'
	opt['mb_size'] = 32
	opt['n_iterations'] = 100000
	opt['lr_schedule'] = [30000, 50000]
	opt['lr'] = 1e-4
	opt['im_size'] = (150,150)
	opt['train_size'] = 240000
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
	opt['summary_path'] = '{:s}/summaries/class_face15train_{:s}_{:s}{:s}'.format(dir_, opt['loss_type'], opt['method'], lw)
	opt['save_path'] = '{:s}/checkpoints/class_face15train_{:s}_{:s}{:s}/model.ckpt'.format(dir_, opt['loss_type'], opt['method'], lw)

	#check and clear directories
	checkFolder(opt['summary_path'])
	checkFolder(opt['save_path'])
	#removeAllFilesInDirectory(opt['summary_path'], '.*')
	#removeAllFilesInDirectory(opt['save_path'], '.*')
	
	# Load data
	train_files = face_loader.get_files(opt['data_folder'])
	x, target, geometry, lighting, d_params, ids = face_loader.get_batches(train_files, True, opt)
	x /= 255.
	target /= 255.

	# Placeholders
	global_step = tf.Variable(0, name='global_step', trainable=False)
	lr = tf.placeholder(tf.float32, [], name='lr')
	is_training = tf.placeholder(tf.bool, [], name='is_training')
	labels = tf.placeholder(tf.int32, [None,], name='labels')
	
	# Build the training model
	zeros = tf.zeros_like(geometry)
	f_params1 = tf.concat([geometry,zeros], 1)
	f_params2 = tf.concat([zeros,lighting], 1)
	f_params = tf.concat([f_params1, f_params2], 2)
	
	if opt['method'] == 'efros':
		recon = autoencoder_efros(x, d_params, is_training, opt)	
	elif opt['method'] == 'worrall':
		recon, latents = autoencoder(x, f_params, is_training, opt)
	elif opt['method'] == 'kulkarni':
		recon = autoencoder_DCIGN(x, f_params, is_training, opt)

	# LOSS
	if opt['loss_type'] == 'xentropy':
		loss = bernoulli_xentropy(target, recon, mean=True)
	elif opt['loss_type'] == 'L2':
		loss = gaussian_nll(target, tf.nn.sigmoid(recon), mean=True)
	elif opt['loss_type'] == 'SSIM':
		loss = SSIM(target, tf.nn.sigmoid(recon), mean=True)
	elif opt['loss_type'] == 'SSIM_L1':
		loss = opt['loss_weights'][0]*SSIM(target, tf.nn.sigmoid(recon), mean=True)
		loss += opt['loss_weights'][1]*mean_loss(tf.abs(target - tf.nn.sigmoid(recon)), mean=True)
	
	y = classifier(latents, opt)
	ids = tf.to_int32(tf.string_to_number(ids))
	class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ids, logits=latents))
	combined_loss = 0.1*class_loss + loss
	acc = tf.reduce_mean(tf.to_float(tf.equal(ids, tf.to_int32(tf.argmax(latents, axis=1)))))
	
	# Summaries
	tf.summary.scalar('ReconstructionLoss', loss)
	tf.summary.scalar('CombinedLoss', combined_loss)
	tf.summary.scalar('LearningRate', lr)
	tf.summary.scalar('ClassificationLoss', class_loss)
	tf.summary.scalar('Accuracy', acc)
	merged = tf.summary.merge_all()
	input_summary = tf.summary.image('Inputs', x, max_outputs=3)
	target_summary = tf.summary.image('Targets', target, max_outputs=3)
	recon_summary = tf.summary.image('Reconstruction', tf.nn.sigmoid(recon), max_outputs=3)
	
	# Build optimizer
	optim = tf.train.AdamOptimizer(lr)
	train_op = optim.minimize(loss, global_step=global_step)

	# Set inputs, outputs, and training ops
	inputs = [lr, is_training]
	outputs = [combined_loss]
	summaries = [merged, recon_summary, target_summary, input_summary]
	ops = [train_op, global_step]
	
	# Train
	return train(inputs, outputs, ops, summaries, opt)

if __name__ == '__main__':
	tf.app.run()


























