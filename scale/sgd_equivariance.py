'''SGD equivariance'''

import os
import sys
import time
sys.path.append('../')

import cv2
import input_data
import numpy as np
import skimage.io as skio
import tensorflow as tf

from matplotlib import pyplot as plt
from spatial_transformer import transformer


def load_data():
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
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


def train(inputs, outputs, optim_step, opt):
	"""Training loop"""
	x, labels, t_params, f_params, lr = inputs
	loss, acc, y, yr = outputs
	
	data = load_data()
	n_data = data['X']['train'].shape[0]
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		feed_dict = {x: data['X']['train'][:opt['mb_size'],...],
						 t_params: np.zeros((opt['mb_size'],6))}
		sess.run(init, feed_dict=feed_dict)
		
		for epoch in xrange(opt['n_epochs']):
			loss_total = 0.
			acc_total = 0.
			mb_list = random_sampler(n_data, opt)
			current_lr = opt['lr']*np.power(0.1, epoch/opt['lr_interval'])
			for i, mb in enumerate(mb_list):
				tp, fp = random_transform(mb.shape[0], (28,28))
				ops = [loss, acc, optim_step]
				feed_dict = {x: data['X']['train'][mb,...],
								 labels: data['Y']['train'][mb,...],
								 t_params: tp,
								 f_params: fp,
								 lr: current_lr}
				l, c, __ = sess.run(ops, feed_dict=feed_dict)
				loss_total += l
				acc_total += c
				sys.stdout.write('{:03d}%\r'.format(int((100.*i)/len(mb_list))))
				sys.stdout.flush()
			loss_total = loss_total / (i+1.)
			acc_total = acc_total / (i+1.)
		
			vacc_total = 0.
			mb_list = random_sampler(data['Y']['valid'].shape[0], opt, random=False)
			for i, mb in enumerate(mb_list):
				feed_dict = {x: data['X']['valid'][mb,...],
								 labels: data['Y']['valid'][mb,...]}
				vc = sess.run(acc, feed_dict=feed_dict)
				vacc_total += vc
			vacc_total = vacc_total / (i+1.)
			
			print('[{:04d}]: Loss: {:04f}, Train Acc.: {:04f}, Valid Acc.: {:04f}' \
					.format(epoch, loss_total, acc_total, vacc_total))
	
		tacc_total = 0.
		mb_list = random_sampler(data['Y']['test'].shape[0], opt, random=False)
		for i, mb in enumerate(mb_list):
			feed_dict = {x: data['X']['test'][mb,...],
							 labels: data['Y']['test'][mb,...]}
			tc = sess.run(acc, feed_dict=feed_dict)
			tacc_total += tc
		tacc_total = tacc_total / (i+1.)
		
		print('Test Acc.: {:04f}'.format(tacc_total))
	return tacc_total

def conv(x, shape, name='0', bias_init=0.01, return_params=False):
	"""Basic convolution"""
	stddev = tf.sqrt(1./tf.reduce_sum(tf.to_float(shape[1:])))
	He_initializer = tf.random_normal_initializer(stddev=stddev)
	W = tf.get_variable('W'+name, shape=shape, initializer=He_initializer)
	z = tf.nn.conv2d(x, W, (1,1,1,1), 'SAME', name='conv'+str(name))
	return z


def bias_add(x, nc, bias_init=0.01, name='0'):
	const_initializer = tf.constant_initializer(value=bias_init)
	b = tf.get_variable('b'+name, shape=nc, initializer=const_initializer)
	return tf.nn.bias_add(x, b)


def S_nonlin(x, sh, nc, fnc=tf.nn.relu, eps=1e-12, name='0', device='/gpu:0'):
	"""Nonlinearities defined on the circle"""
	x = tf.reshape(x, (-1,sh,sh,2,nc/2))
	R = tf.reduce_sum(tf.square(x), reduction_indices=[3], keep_dims=True)
	magnitude = tf.sqrt(tf.maximum(R,eps))
	with tf.device(device):
		const_initializer = tf.constant_initializer(value=-0.05)
		b = tf.get_variable('b'+name, initializer=const_initializer, shape=[nc/2])
	
	Rb = tf.nn.bias_add(magnitude, b)
	c = tf.div(fnc(Rb), magnitude)
	return tf.reshape(c*x, (-1,sh,sh,nc))


def siamese_model(x, t_params, f_params, opt):
	"""Build siamese models for equivariance tests"""
	nc = opt['n_channels']
	xsh = x.get_shape().as_list()
	# Mouth
	with tf.variable_scope('mouth') as scope:
		y1 = mouth_ception(x, nc, opt, name='_MC')
		# Transformer branch
		y_post = transform_features(y1, t_params, f_params)
		
		# Siamese loss
		x_pre = transformer(x, t_params, (xsh[1],xsh[2]))
		scope.reuse_variables()
		y2 = mouth_ception(x_pre, nc, opt, name='_MC')
		
		# Tail
	with tf.variable_scope('tail') as scope:
		y = tf.nn.max_pool(y1, (1,2,2,1), (1,2,2,1), padding='VALID')
		logits = build_tail(y, 3*nc, opt, name='tail')
	
	return logits, y_post, y2


def mouth_ception(x, nc, opt, name='_MC'):
	# L1
	l1 = build_mouth(x, nc, opt, name='_M1'+name)
	l1 = tf.nn.max_pool(l1, (1,2,2,1), (1,2,2,1), padding='VALID')
	# L2
	return build_mouth(l1, 2*nc, opt, name='_M2'+name)


def build_mouth(x, nc, opt, name='mouth'):
	"""Build the model we want"""
	xsh = x.get_shape().as_list()
	l1 = conv(x, [3,3,xsh[3],nc], name='1'+name )
	l1 = bias_add(l1, nc, name='1'+name)
	l1 = tf.nn.relu(l1)
	
	l2 = conv(l1, [3,3,nc,nc], name='2'+name)
	l2 = bias_add(l2, nc, name='2'+name)
	return l2


def build_tail(x, nc, opt, name='tail'):
	"""Build the model we want"""
	xsh = x.get_shape().as_list()
	l1 = conv(x, [3,3,xsh[3],nc], name='1'+name )
	l1 = bias_add(l1, nc, name='1'+name)
	l1 = tf.nn.relu(l1)
	
	l2 = conv(l1, [3,3,nc,10], name='2'+name)
	l2 = bias_add(l2, 10, name='2'+name)
	l2 = tf.reduce_mean(l2, reduction_indices=(1,2))
	return l2


def transform_features(x, t_params, f_params):
	"""Rotate features in the channels"""
	# 1) Rotate features through channels. We have to perform a broadcasted
	# matrix--matrix multiply on two subarrays of the whole tensor, but this does
	# not currently exist in TensorFlow, so we have to do it the long way.
	xsh = x.get_shape().as_list()
	x = tf.reshape(x, tf.pack([xsh[0],xsh[1],xsh[2],xsh[3]/2,2]))
	f1 = tf.reshape(f_params[:,0,:], tf.pack([xsh[0],1,1,1,2]))
	f2 = tf.reshape(f_params[:,1,:], tf.pack([xsh[0],1,1,1,2]))
	x0 = tf.reduce_sum(tf.mul(x, f1), reduction_indices=4)
	x1 = tf.reduce_sum(tf.mul(x, f2), reduction_indices=4)
	x = tf.pack([x0, x1], axis=-1)
	x = tf.reshape(x, tf.pack([xsh[0],xsh[1],xsh[2],xsh[3]]))
	# 2) Rotate features spatially
	y = transformer(x, t_params, (xsh[1],xsh[2]))
	return y


def get_t_transform(theta, imsh):
	scale1 = np.array([[float(imsh[0])/imsh[1], 0.], [0., 1.]])
	scale2 = np.array([[float(imsh[1])/imsh[0], 0.], [0., 1.]])
	rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	mat = np.dot(scale1, rot)
	mat = np.dot(mat, scale2)
	
	mat = np.hstack((mat,np.zeros((2,1))))
	mat = mat.astype('float32')
	mat = mat.flatten()
	return mat


def get_f_transform(theta):
	rot = np.array([[np.cos(theta), -np.sin(theta)],
						[np.sin(theta), np.cos(theta)]])
	return rot


def random_transform(mb_size, imsh):
	t_params = []
	f_params = []
	for t in np.random.rand(mb_size):
		t_params.append(get_t_transform(2*np.pi*t, (imsh[0],imsh[1])))
		f_params.append(get_f_transform(2*np.pi*t))
	return np.vstack(t_params), np.stack(f_params, axis=0)

	
def main(opt):
	"""Main loop"""
	
	tf.reset_default_graph()
	opt['N'] = 28
	opt['mb_size'] = 128
	opt['n_channels'] = 10
	opt['n_epochs'] = 100
	opt['lr'] = 1e-2
	opt['lr_interval'] = 33
	#opt['equivariant_weight'] = 1e-3
	
	# Define variables
	x = tf.placeholder(tf.float32, [opt['mb_size'],opt['N'],opt['N'],1], name='x')
	labels = tf.placeholder(tf.float32, [opt['mb_size'],10], name='labels')
	t_params = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
	f_params = tf.placeholder(tf.float32, [opt['mb_size'],2,2], name='f_params')
	
	lr = tf.placeholder(tf.float32, [], name='lr')
	logits, y, yr = siamese_model(x, t_params, f_params, opt)

	# Build loss and metrics
	equivariant_loss = tf.reduce_mean(tf.square(y - yr))
	classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
	loss = classification_loss + opt['equivariant_weight']*equivariant_loss
	
	acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), tf.float32))
	
	# Build optimizer
	optim = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
	optim_step = optim.minimize(loss)
	
	inputs = [x, labels, t_params, f_params, lr]
	outputs = [loss, acc, y, yr]
	
	# Train
	return train(inputs, outputs, optim_step, opt)


if __name__ == '__main__':
	opt = {}
	main(opt)
