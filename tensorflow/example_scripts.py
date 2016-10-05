'''MNIST tests'''

'''Test the gConv script'''

import os
import sys
import time

import cv2
import numpy as np
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import tensorflow as tf

import input_data

from rotated_conv import *
from matplotlib import pyplot as plt

##### MODELS #####

def conv_Z(x, drop_prob, n_filters, n_classes):
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([3,3,1,n_filters], name='W1'),
		'w2' : get_weights([3,3,n_filters,n_filters], name='W2'),
		'w3' : get_weights([n_filters*5*5,500], name='W3'),
		'out': get_weights([500, n_classes], name='W4')
	}
	
	biases = {
		'b1': tf.Variable(tf.constant(1e-2, shape=[n_filters])),
		'b2': tf.Variable(tf.constant(1e-2, shape=[n_filters])),
		'b3': tf.Variable(tf.constant(1e-2, shape=[500])),
		'out': tf.Variable(tf.constant(1e-2, shape=[n_classes]))
	}
	
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	# Convolution Layer
	cv1 = conv2d(x, weights['w1'], biases['b1'], name='gc1')
	mp1 = tf.nn.relu(maxpool2d(cv1, k=2))
	
	# Convolution Layer
	cv2 = conv2d(mp1, weights['w2'], biases['b2'], name='gc2')
	mp2 = tf.nn.relu(maxpool2d(cv2, k=2))

	# Fully connected layer
	fc3 = tf.reshape(mp2, [-1, weights['w3'].get_shape().as_list()[0]])
	fc3 = tf.nn.bias_add(tf.matmul(fc3, weights['w3']), biases['b3'])
	fc3 = tf.nn.relu(fc3)
	# Apply Dropout
	fc3 = tf.nn.dropout(fc3, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc3, weights['out']), biases['out'])
	return out

def gConv_real_steer(x, drop_prob, n_filters, n_classes, bs, phase_train):
	# Store layers weight & bias
	weights = {
		'w1' : [get_weights([3,1,n_filters], name='W1_0'),
				get_weights([2,1,n_filters], name='W1_1'),
				get_weights([2,1,n_filters], name='W1_2'),
				get_weights([2,1,n_filters], name='W1_3'),
				get_weights([2,1,n_filters], name='W1_4')],
		'w2' : get_weights([1,1,2*n_filters,n_filters], name='W2'),
		'w3' : get_weights([2*n_filters*12*12,500], name='W3'),
		'out': get_weights([500, n_classes], name='W4')
	}
	
	biases = {
		'b1': tf.Variable(tf.constant(1e-2, shape=[n_filters])),
		'b2': tf.Variable(tf.constant(1e-2, shape=[n_filters])),
		'b3': tf.Variable(tf.constant(1e-2, shape=[500])),
		'out': tf.Variable(tf.constant(1e-2, shape=[n_classes]))
	}
	nonlinearity = complex_softplus
	
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolution Layer
	re1 = equi_real_conv(x, weights['w1'])
	re1 = nonlinearity(re1, biases['b1'])
	re1 = re1[1:3]

	# Convolution Layer
	re2 = complex_steer_conv(re1, weights['w2'], strides=(1,2,2,1))
	re2 = nonlinearity(re2, biases['b2'])

	# Fully connected layer
	R = tf.concat(3, re2)
	
	fc3 = tf.reshape(R, [bs, weights['w3'].get_shape().as_list()[0]])
	fc3 = tf.nn.bias_add(tf.matmul(fc3, weights['w3']), biases['b3'])
	fc3 = tf.nn.relu(fc3)
	fc3 = tf.nn.dropout(fc3, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc3, weights['out']), biases['out'])
	return out

def phase_discard(x, drop_prob, n_filters, n_classes, bs, bn_config, phase_train):
	# Store layers weight & bias
	weights = {
		'w5' : get_weights([n_filters*7*7,500], name='W5'),
		'out': get_weights([500, n_classes], name='W6')
	}
	
	biases = {
		'b5': tf.Variable(tf.constant(1e-2, shape=[500])),
		'out': tf.Variable(tf.constant(1e-2, shape=[n_classes]))
	}
	order = 3
	bn = bn_config
	nf = n_filters
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolution Layer
	rb1 = residual_block(x, 1, nf, order, phase_train, pool_in=False, bn=bn[0], name='rb1')
	rb2 = residual_block(rb1, nf, nf, order, phase_train, pool_in=True, bn=bn[1], name='rb2')
		
	# Fully connected layer
	rb2 = maxpool2d(rb2)
	
	fc = tf.reshape(tf.nn.dropout(rb2, drop_prob), [bs, weights['w5'].get_shape().as_list()[0]])
	fc = tf.nn.bias_add(tf.matmul(fc, weights['w5']), biases['b5'])
	fc = tf.nn.relu(fc)
	fc = tf.nn.dropout(fc, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc, weights['out']), biases['out'])
	return out



##### CUSTOM BLOCKS #####
def residual_block(x, n_in, n_out, order, phase_train, pool_in=True, bn=True, name='rb'):
	W1 = get_weights_list([3]+order*[2], n_in, n_out, name=name+'W1')
	W2 = get_weights_list([3]+order*[2], n_out, n_out, name=name+'W2')
	b1 = get_bias_list(n_out, order, name=name+'b1')
	b2 = get_bias_list(n_out, order, name='b2')
	
	if pool_in:
		x = maxpool2d(x)
	re1 = equi_real_conv(x, W1, order=order, padding='SAME')
	re1 = phase_invariant_relu(re1, b1, order=order)
	re1 = tf.add_n(re1)
		
	re2 = equi_real_conv(re1, W2, order=order, padding='SAME')
	re2 = phase_invariant_relu(re2, b2, order=order)
	re2 = tf.add_n(re2)
	re2 = re2 + x
	if bn:
		re2 = batch_norm(re2, n_out, phase_train)
	return re2

def conv2d(X, V, b=None, strides=(1,1,1,1), padding='VALID', name='conv2d'):
    """conv2d wrapper. Supply input X, weights V and optional bias"""
    VX = tf.nn.conv2d(X, V, strides=strides, padding=padding, name=name+'_')
    if b is not None:
        VX = tf.nn.bias_add(VX, b)
    return VX

def maxpool2d(X, k=2):
    """Tied max pool. k is the stride and pool size"""
    return tf.nn.max_pool(X, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def get_weights(filter_shape, W_init=None, name='W'):
	"""Initialize weights variable with Xavier method"""
	if W_init == None:
		stddev = np.sqrt(2.0 / np.prod(filter_shape[:2]))
		W_init = tf.random_normal(filter_shape, stddev=stddev)
	return tf.Variable(W_init, name=name)

def get_weights_list(comp_shape, in_shape, out_shape, name='W'):
	"""Return a list of weights for use with equi_real_conv(). comp_shape is a
	list of the number of elements per Fourier base. For 3x3 weights use
	[3,2,2,2]. I'm going to change this to just accept 'order' and kernel size
	in future."""
	weights_list = []
	for i, cs in enumerate(comp_shape):
		shape = [cs,in_shape,out_shape]
		weights_list.append(get_weights(shape, name=name+'_'+str(i)))
	return weights_list

def get_bias_list(n_filters, order, name='b'):
	"""Return a list of biases for use with equi_real_conv()"""
	bias_list = []
	for i in xrange(order+1):
		bias = tf.Variable(tf.constant(1e-2, shape=[n_filters]), name=name+'_'+str(i))
		bias_list.append(bias)
	return bias_list

def minibatcher(inputs, targets, batch_size, shuffle=False):
	"""Input and target are minibatched. Returns a generator"""
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield inputs[excerpt], targets[excerpt]


##### MAIN SCRIPT #####
def run(model='deep_steer', lr=1e-2, batch_size=250, n_epochs=500, n_filters=30,
		bn_config=[False, False], trial_num='N', combine_train_val=False):
	
	# Load dataset
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')
	mnist_trainx, mnist_trainy = mnist_train['x'], mnist_train['y']
	mnist_validx, mnist_validy = mnist_valid['x'], mnist_valid['y']
	mnist_testx, mnist_testy = mnist_test['x'], mnist_test['y']

	# Parameters
	lr = lr
	batch_size = batch_size
	n_epochs = n_epochs
	save_step = 100		# Not used yet
	model = model
	
	# Network Parameters
	n_input = 784 	# MNIST data input (img shape: 28*28)
	n_classes = 10 	# MNIST total classes (0-9 digits)
	dropout = 0.75 	# Dropout, probability to keep units
	n_filters = n_filters
	dataset_size = 10000
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	
	# Construct model
	if model == 'conv_Z':
		pred = conv_Z(x, keep_prob, n_filters, n_classes)
	elif model == 'gConv_real_steer':
		pred = gConv_real_steer(x, keep_prob, n_filters, n_classes, batch_size, phase_train)
	elif model == 'phase_discard':
		pred= phase_discard(x, keep_prob, n_filters, n_classes, batch_size, bn_config, phase_train)
	else:
		print('Model unrecognized')
		sys.exit(1)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	if combine_train_val:
		mnist_trainx = np.vstack([mnist_trainx, mnist_validx])
		mnist_trainy = np.hstack([mnist_trainy, mnist_validy])
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		epoch = 0
		start = time.time()
		# Keep training until reach max iterations
		while epoch < n_epochs:
			generator = minibatcher(mnist_trainx, mnist_trainy, batch_size, shuffle=True)
			cost_total = 0.
			acc_total = 0.
			vacc_total = 0.
			for i, batch in enumerate(generator):
				batch_x, batch_y = batch
				lr_current = lr/np.sqrt(1.+epoch*(float(batch_size) / dataset_size))
				
				# Optimize
				feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout,
							 learning_rate : lr_current, phase_train : True}
				__, cost_, acc_ = sess.run([optimizer, cost, accuracy], feed_dict=feed_dict)
				cost_total += cost_
				acc_total += acc_
			cost_total /=(i+1.)
			acc_total /=(i+1.)
			
			if not combine_train_val:
				val_generator = minibatcher(mnist_validx, mnist_validy, batch_size, shuffle=False)
				for i, batch in enumerate(val_generator):
					batch_x, batch_y = batch
					# Calculate batch loss and accuracy
					feed_dict = {x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
					vacc_ = sess.run(accuracy, feed_dict=feed_dict)
					vacc_total += vacc_
				vacc_total = vacc_total/(i+1.)		

			print "[" + str(trial_num),str(epoch) + \
				"], Minibatch Loss: " + \
				"{:.6f}".format(cost_total) + ", Train Acc: " + \
				"{:.5f}".format(acc_total) + ", Time: " + \
				"{:.5f}".format(time.time()-start) + ", Val acc: " + \
				"{:.5f}".format(vacc_total)
			epoch += 1
		
		print "Testing"
		
		# Test accuracy
		tacc_total = 0.
		test_generator = minibatcher(mnist_testx, mnist_testy, batch_size, shuffle=False)
		for i, batch in enumerate(test_generator):
			batch_x, batch_y = batch
			feed_dict={x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
			tacc = sess.run(accuracy, feed_dict=feed_dict)
			tacc_total += tacc
		tacc_total = tacc_total/(i+1.)
		print('Test accuracy: %f' % (tacc_total,))
	return tacc_total



if __name__ == '__main__':
	run(model='phase_discard', lr=1e-4, batch_size=132, n_epochs=500,
		n_filters=15, combine_train_val=False)
