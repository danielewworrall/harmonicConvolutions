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

from gConv2 import *
from helpers import *
from matplotlib import pyplot as plt


# Create model
def conv_Z(x, drop_prob, n_filters, n_classes):
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([3,3,1,n_filters], name='W1'),
		'w2' : get_weights([3,3,n_filters,n_filters], name='W2'),
		'w3' : get_weights([n_filters*6*6,500], name='W3'),
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

def gConv_simple(x, drop_prob, n_filters, n_classes):
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([3,3,1,n_filters], name='W1'),
		'w2' : get_weights([3,3,n_filters*4,n_filters], name='W2'),
		'w3' : get_weights([n_filters*6*6,500], name='W3'),
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
	gc1 = gConv(x, weights['w1'], biases['b1'], name='gc1')
	gc1_ = tf.nn.relu(maxpool2d(gc1, k=2))
	
	# Convolution Layer
	gc2 = gConv(gc1_, weights['w2'], biases['b2'], name='gc2')
	gc2_ = coset_pooling(gc2)
	gc2_ = tf.nn.relu(maxpool2d(gc2_, k=2))

	# Fully connected layer
	fc3 = tf.reshape(gc2_, [-1, weights['w3'].get_shape().as_list()[0]])
	fc3 = tf.nn.bias_add(tf.matmul(fc3, weights['w3']), biases['b3'])
	fc3 = tf.nn.relu(fc3)
	# Apply Dropout
	fc3 = tf.nn.dropout(fc3, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc3, weights['out']), biases['out'])
	return out

def gConv_taco(x, drop_prob, n_filters, n_classes):
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([3,3,1,n_filters], name='W1'),
		'w2' : get_weights([3,3,n_filters*4,n_filters], name='W2'),
		'w3' : get_weights([3,3,n_filters*4,n_filters], name='W3'),
		'w4' : get_weights([3,3,n_filters*4,n_filters], name='W4'),
		'w5' : get_weights([3,3,n_filters*4,n_filters], name='W5'),
		'w6' : get_weights([3,3,n_filters*4,n_filters], name='W6'),
		'w7' : get_weights([3,3,n_filters*4,n_filters], name='W7'),
		'out': get_weights([160, n_classes], name='W4')
	}
	
	biases = {
		'out': tf.Variable(tf.constant(1e-2, shape=[n_classes]))
	}
	
	# Reshape input picture
	xin = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	# Convolution Layerss
	gc1 = tf.nn.relu(gConv(xin, weights['w1'], name='gc1'))
	gc2 = tf.nn.relu(gConv(gc1, weights['w2'], name='gc2'))
	p2_ = maxpool2d(gc2, k=2)
	gc3 = tf.nn.relu(gConv(p2_, weights['w3'], name='gc3'))
	gc4 = tf.nn.relu(gConv(gc3, weights['w4'], name='gc4'))
	gc5 = tf.nn.relu(gConv(gc4, weights['w5'], name='gc5'))
	gc6 = tf.nn.relu(gConv(gc5, weights['w6'], name='gc6'))
	gc7 = tf.nn.relu(gConv(gc6, weights['w7'], name='gc7'))
	
	gc7_ = coset_pooling(gc7)
	gc7_ = tf.reshape(gc7, [-1,160])
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(gc7_, weights['out']), biases['out'])
	return out

def gConv_steer(x, drop_prob, n_filters, n_classes):
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([1,1,2,n_filters], name='W1'),
		'w2' : get_weights([1,1,2*n_filters,n_filters], name='W2'),
		'w3' : get_weights([n_filters*12*12,500], name='W3'),
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
	sc1 = steer_conv(x, weights['w1'], biases['b1'])
	#mp1 = tf.nn.relu(maxpool2d(sc1, k=2))
	mp1 = tf.nn.relu(sc1)
	
	# Convolution Layer
	sc2 = steer_conv(mp1, weights['w2'], biases['b2'], strides=(1,2,2,1))
	#mp2 = tf.nn.relu(maxpool2d(sc2, k=2))
	mp2 = tf.nn.relu(sc2)

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
		'w1' : get_weights([3,1,n_filters], name='W1'),
		'w2' : get_weights([1,1,2*n_filters,n_filters], name='W2'),
		'w3' : get_weights([n_filters*12*12,500], name='W3'),
		'out': get_weights([500, n_classes], name='W4')
	}
	
	biases = {
		'b1': tf.Variable(tf.constant(1e-2, shape=[n_filters])),
		'b2': tf.Variable(tf.constant(1e-2, shape=[n_filters])),
		'b3': tf.Variable(tf.constant(1e-2, shape=[500])),
		'out': tf.Variable(tf.constant(1e-2, shape=[n_classes]))
	}
	
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolution Layer
	re1 = equi_real_conv(x, weights['w1'])
	re1 = complex_batch_norm(re1, phase_train)
	re1 = complex_relu(re1, biases['b1'])

	# Convolution Layer
	re2 = complex_steer_conv(re1, weights['w2'], strides=(1,2,2,1))
	re2 = complex_batch_norm(re2, phase_train)
	re2 = complex_relu(re2, biases['b2'])

	# Fully connected layer
	nlx, nly = re2
	R = nlx
	
	fc3 = tf.reshape(R, [bs, weights['w3'].get_shape().as_list()[0]])
	fc3 = tf.nn.bias_add(tf.matmul(fc3, weights['w3']), biases['b3'])
	fc3 = tf.nn.relu(fc3)
	fc3 = tf.nn.dropout(fc3, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc3, weights['out']), biases['out'])
	return out

def deep_Z(x, n_filters, n_classes, bs, phase_train):
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([3,3,1,n_filters], name='W1'),
		'w2' : get_weights([3,3,n_filters,n_filters], name='W2'),
		'w3' : get_weights([3,3,n_filters,n_filters], name='W3'),
		'w4' : get_weights([3,3,n_filters,n_filters], name='W3'),
		'w5' : get_weights([3,3,n_filters,n_filters], name='W3'),
		'w6' : get_weights([3,3,n_filters,n_filters], name='W3'),
		'w7' : get_weights([4,4,n_filters,n_classes], name='W3'),
	}
	
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolution Layer
	rc1 = conv2d(x, weights['w1'])
	rc1 = batch_norm(rc1, n_filters, phase_train)
	rc2 = conv2d(tf.nn.relu(rc1), weights['w2'])
	rc2 = batch_norm(rc2, n_filters, phase_train)
	
	rc2 = maxpool2d(rc2)
	rc3 = conv2d(tf.nn.relu(rc2), weights['w3'])
	rc3 = batch_norm(rc3, n_filters, phase_train)
	rc4 = conv2d(tf.nn.relu(rc3), weights['w4'])
	rc4 = batch_norm(rc4, n_filters, phase_train)

	rc5 = conv2d(tf.nn.relu(rc4), weights['w5'])
	rc5 = batch_norm(rc5, n_filters, phase_train)
	rc6 = conv2d(tf.nn.relu(rc5), weights['w6'])
	rc6 = batch_norm(rc6, n_filters, phase_train)
	
	rc7 = conv2d(tf.nn.relu(rc6), weights['w7'])
	
	return tf.reduce_sum(rc7, reduction_indices=[1,2])

def deep_steer(x, n_filters, n_classes, bs, phase_train, eps=1e-2):
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([3,1,n_filters], name='W1'),
		'w2' : get_weights([1,1,2*n_filters,n_filters], name='W2'),
		'w3' : get_weights([1,1,2*n_filters,n_filters], name='W3'),
		'w4' : get_weights([1,1,2*n_filters,n_filters], name='W4'),
		'w5' : get_weights([1,1,2*n_filters,n_filters], name='W5'),
		'w6' : get_weights([1,1,2*n_filters,n_filters], name='W6'),
		'w7' : get_weights([1,1,2*n_filters,n_classes], name='W7'),
	}
	
	biases = {
		'b1': tf.Variable(tf.constant(1e-2, shape=[n_filters]), name='b1'),
		'b2': tf.Variable(tf.constant(1e-2, shape=[n_filters]), name='b2'),
		'b3': tf.Variable(tf.constant(1e-2, shape=[n_filters]), name='b3'),
		'b4': tf.Variable(tf.constant(1e-2, shape=[n_filters]), name='b4'),
		'b5': tf.Variable(tf.constant(1e-2, shape=[n_filters]), name='b5'),
		'b6': tf.Variable(tf.constant(1e-2, shape=[n_filters]), name='b6'),
		'b7': tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='b7'),
	}
	
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolution Layer
	rc1 = equi_real_conv(x, weights['w1'])
	#rc1 = complex_batch_norm(rc1, phase_train)
	rc1 = complex_relu(rc1, biases['b1'])
	
	cc2 = complex_steer_conv(rc1, weights['w2'])
	#cc2 = complex_batch_norm(cc2, phase_train)
	cc2 = complex_relu(cc2, biases['b2'])
	
	
	cc3 = complex_steer_conv(cc2, weights['w3'], strides=(1,2,2,1))
	#cc3 = complex_batch_norm(cc3, phase_train)
	cc3 = complex_relu(cc3, biases['b3'])
	
	cc4 = complex_steer_conv(cc3, weights['w4'])
	#cc4 = complex_batch_norm(cc4, phase_train)
	cc4 = complex_relu(cc4, biases['b4'])
	
	cc5 = complex_steer_conv(cc4, weights['w5'])	
	#cc5 = complex_batch_norm(cc5, phase_train)
	cc5 = complex_relu(cc5, biases['b5'])
	
	cc6 = complex_steer_conv(cc5, weights['w6'])
	#cc6 = complex_batch_norm(cc6, phase_train)
	cc6 = complex_relu(cc6, biases['b6'])
	
	
	cc7 = complex_steer_conv(cc6, weights['w7'])
	nlx, nly = cc7
	
	R = tf.sqrt(tf.square(nlx) + tf.square(nly) + eps)
	y = tf.nn.bias_add(tf.reduce_sum(R, reduction_indices=[1,2]), biases['b7'])
	return y

def single_steer(x, n_filters, n_classes, bs, phase_train, eps=1e-2):
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([3,1,n_filters], name='W1')
	}
	
	biases = {
		'b1': tf.Variable(tf.constant(1e-2, shape=[n_filters]), name='b1')
	}
	
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolution Layer
	rc1 = equi_real_conv(x, weights['w1'])
	rc1 = complex_batch_norm(rc1, phase_train)
	rc1 = complex_relu(rc1, biases['b1'])
	
	re, im = rc1
	return re, im

def conv2d(X, V, b=None, strides=(1,1,1,1), padding='VALID', name='conv2d'):
    """conv2d wrapper"""
    VX = tf.nn.conv2d(X, V, strides=strides, padding=padding, name=name+'_')
    if b is not None:
        VX = tf.nn.bias_add(VX, b)
    return VX

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def get_weights(filter_shape, W_init=None, name='W'):
	if W_init == None:
		stddev = np.sqrt(2.0 / np.prod(filter_shape[:2]))
		W_init = tf.random_normal(filter_shape, stddev=stddev)
	return tf.Variable(W_init, name=name)

def random_rotation(X):
	"""Randomly rotate images, independently in minibatch"""
	X = np.reshape(X, [-1,28,28,1])
	Xsh = X.shape
	for i in xrange(Xsh[0]):
		angle = np.random.rand() * 360.
		M = cv2.getRotationMatrix2D((Xsh[2]/2,Xsh[1]/2), angle, 1)
		X[i,...] = cv2.warpAffine(X[i,...], M, (Xsh[2],Xsh[1])).reshape(Xsh[1:])
	return X.reshape(-1,784)

def ring_rotation(X, n=50):
	"""Rotate images, along ring"""
	X = np.reshape(X, [28,28,1])
	X_ = np.zeros((n,28,28,1))
	Xsh = X.shape
	for i in xrange(n):
		angle = (360.*i)/n
		M = cv2.getRotationMatrix2D((Xsh[1]/2,Xsh[0]/2), angle, 1)
		X_[i,...] = cv2.warpAffine(X, M, (Xsh[1],Xsh[0])).reshape(Xsh)
	return X_.reshape(-1,784)

def run():
	#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')

	# Parameters
	lr = 1e-2
	batch_size = 250
	dataset_size = 50000
	valid_size = 5000
	n_epochs = 500
	display_step = dataset_size / batch_size
	save_step = 100
	model = 'deep_steer'
	test_rot = True
	
	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)
	dropout = 0.75 # Dropout, probability to keep units
	n_filters = 20
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	
	# Construct model
	if model == 'simple_Z':
		pred = conv_Z(x, keep_prob, n_filters, n_classes)
	elif model == 'simple':
		pred = gConv_simple(x, keep_prob, n_filters, n_classes)
	elif model == 'taco':
		pred = gConv_taco(x, keep_prob, n_filters, n_classes)
	elif model == 'steer':
		pred = gConv_steer(x, keep_prob, n_filters, n_classes)
	elif model == 'equi_steer':
		pred = gConv_real_steer(x, keep_prob, n_filters, n_classes, batch_size, phase_train)
	elif model == 'deep_steer':
		pred = deep_steer(x, n_filters, n_classes, batch_size, phase_train)
	elif model == 'deep_Z':
		pred = deep_Z(x, n_filters, n_classes, batch_size, phase_train)
	else:
		print('Model unrecognized')
		sys.exit(1)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
	#optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		epoch = 0
		start = time.time()
		# Keep training until reach max iterations
		while epoch < n_epochs:
			generator = minibatcher(mnist_train['x'], mnist_train['y'], batch_size, shuffle=True)
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
			
			val_generator = minibatcher(mnist_valid['x'], mnist_valid['y'], batch_size, shuffle=False)
			for i, batch in enumerate(val_generator):
				batch_x, batch_y = batch
				# Calculate batch loss and accuracy
				feed_dict = {x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
				vacc_ = sess.run(accuracy, feed_dict=feed_dict)
				vacc_total += vacc_
			vacc_total = vacc_total/(i+1.)		

			print "[" + str(epoch) + \
				"], Minibatch Loss: " + \
				"{:.6f}".format(cost_total) + ", Train Acc: " + \
				"{:.5f}".format(acc_total) + ", Time: " + \
				"{:.5f}".format(time.time()-start) + ", Val acc: " + \
				"{:.5f}".format(vacc_total)
			epoch += 1
		
		print "Testing"
		
		# Test accuracy
		tacc_total = 0.
		test_generator = minibatcher(mnist_test['x'], mnist_test['y'], batch_size, shuffle=False)
		for i, batch in enumerate(test_generator):
			batch_x, batch_y = batch
			feed_dict={x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
			tacc = sess.run(accuracy, feed_dict=feed_dict)
			tacc_total += tacc
		tacc_total = tacc_total/(i+1.)
		print('Test accuracy: %f' % (tacc_total,))


def forward():
	"""Experiment to demonstrate the equivariance of the convolution"""
	#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')

	# Parameters
	lr = 1e-2
	batch_size = 500
	dataset_size = 50000
	valid_size = 5000
	n_epochs = 150
	display_epoch = 10
	save_step = 100
	test_rot = True
	
	# Network Parameters
	n_input = 784 		# MNIST data input (img shape: 28*28)
	n_classes = 10		# MNIST total classes (0-9 digits)
	dropout = 0.75 		# Dropout, probability to keep units
	n_filters = 10
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	
	# Construct model
	nlx, nly = single_steer(x, n_filters, n_classes, batch_size, phase_train)
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		epoch = 0
		start = time.time()
		# Keep training until reach max iterations
		generator = minibatcher(mnist_train['x'], mnist_train['y'], batch_size, shuffle=True)
		cost_total = 0.
		acc_total = 0.
		vacc_total = 0.
		
		for i, batch in enumerate(generator):
			batch_x, batch_y = batch
			n=500
			batch_x = ring_rotation(batch_x[4,:], n=n)
			
			feed_dict = {x: batch_x, phase_train : False}
			nlx_, nly_ = sess.run([nlx, nly], feed_dict=feed_dict)
			
			R = np.sqrt(nlx_**2 + nly_**2 + 1e-6)
			nlx_ = nlx_/R
			nly_ = nly_/R
			#R = np.reshape(R, [-1,10,1])
			#nlx_ = np.reshape(nlx_, [-1,10,1])
			#nly_ = np.reshape(nly_, [-1,10,1])
			
			plt.figure(1)
			plt.ion()
			plt.show()
			for j in xrange(n):
				plt.cla()
				#plt.imshow(R[j,...,0], cmap='jet', interpolation='nearest')
				#print nlx_[j,...], nly_[j,...]
				plt.quiver(np.mean(nlx_[j,...,0]), np.mean(nly_[j,...,0]))
				plt.draw()
				raw_input(j)

def real_steer_comparison():
	"""Experiment to demonstrate the angular selectivity of the convolution"""
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	
	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_filters = 10
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None,28,28,1])
	v0 = tf.placeholder(tf.float32, [3,1,1])
	#y = equi_steer_conv_(x, v0)
	z = equi_real_conv(x, v0)

	# Initializing the variables
	init = tf.initialize_all_variables()
	
	X = mnist.train.next_batch(100)[0][1,:]
	X = np.reshape(X, [1,28,28,1])
	X_ = np.fliplr(X).T
	X = np.stack((X, X_))
	X = X.reshape([2,28,28,1])
	V0 = np.random.randn(3,1,1).astype(np.float32)
	#V0 = np.ones((3,1,1)).astype(np.float32)
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		X, Y = sess.run(z, feed_dict={x : X, v0 : V0})
	
	#R_0 = np.sqrt(np.sum(Y[0,...]**2, axis=2))
	#X_0, Y_0 = np.squeeze(Y[0,:,:,0])/R_0, np.squeeze(Y[0,:,:,1])/R_0
	X = np.squeeze(X)
	Y = np.squeeze(Y)
	R = np.sqrt(X**2 + Y**2)
	X, Y = X/R, Y/R
	
	plt.figure(1)
	plt.imshow(R[0], cmap='gray', interpolation='nearest')
	plt.quiver(X[0], Y[0])
	plt.figure(2)
	plt.imshow(R[1], cmap='jet', interpolation='nearest')
	plt.quiver(X[1], Y[1])
	plt.show()
	
def Z_steer_comparison():
	"""Experiment to demonstrate the angular selectivity of the convolution"""
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)	

	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_filters = 10
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None,28,28,1])
	v = tf.placeholder(tf.float32, [3,3,1,1])
	z = tf.nn.conv2d(x, v, strides=(1,1,1,1), padding='VALID')

	# Initializing the variables
	init = tf.initialize_all_variables()
	
	X = mnist.train.next_batch(100)[0][1,:]
	X = np.reshape(X, [1,28,28,1])
	X_ = np.fliplr(X).T
	X = np.stack((X, X_))
	X = X.reshape([2,28,28,1])
	V = np.random.randn(3,3,1,1).astype(np.float32)
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		X = sess.run(z, feed_dict={x : X, v : V})
	
	X = np.squeeze(X)
	X_T = np.flipud(X[1].T)
	
	plt.figure(1)
	plt.imshow(X[0], cmap='gray', interpolation='nearest')
	plt.figure(2)
	plt.imshow(X_T, cmap='gray', interpolation='nearest')
	plt.figure(3)
	plt.imshow(X[0] - X_T, cmap='gray', interpolation='nearest')
	plt.figure(4)
	plt.imshow(np.squeeze(V), cmap='gray', interpolation='nearest')
	plt.show()

def complex_steer_test():
	"""Test the complex convolutional filter"""
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	
	# Network Parameters
	n_input = 784 
	n_filters = 10
	
	# tf Graph input
	N = 50
	x = tf.placeholder(tf.float32, [N,28,28,1])
	v0 = tf.placeholder(tf.float32, [1,1,2*1,3])
	b0 = tf.placeholder(tf.float32, [3,])
	v1 = tf.placeholder(tf.float32, [1,1,2*3,1])
	
	y = equi_steer_conv(x, v0)
	#mp = complex_maxpool2d(y, k=2)	# For now process R independently everywhere
	y = complex_relu(y, b0)
	z = complex_steer_conv(y, v1)

	# Initializing the variables
	init = tf.initialize_all_variables()
	
	X = mnist.train.next_batch(100)[0][1,:]
	X = np.reshape(X, [28,28])
	X_ = []
	
	for i in xrange(N):
		angle = i*(360./N)
		X_.append(sciint.rotate(X, angle, reshape=False))
	X = np.reshape(np.stack(X_), [N,28,28,1])
	
	V0 = np.random.randn(1,1,2*1,3).astype(np.float32)
	B0 = np.random.randn(3).astype(np.float32)-0.5
	V1 = np.random.randn(1,1,2*3,1).astype(np.float32)
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		Z = sess.run(z, feed_dict={x : X, v0 : V0, b0 : B0, v1 : V1})
		
	Zx, Zy = Z
	R = np.sqrt(Zx**2 + Zy**2)
	Zx = np.squeeze(Zx/R)
	Zy = np.squeeze(Zy/R)
	R = np.squeeze(R)
	
	plt.figure(1)
	plt.ion()
	plt.show()
	for i in xrange(N):
		plt.cla()
		plt.imshow(R[i], cmap='jet', interpolation='nearest')
		plt.quiver(Zx[i], Zy[i])
		plt.show()
		raw_input(i*(360./N))
	
def complex_small_patch_test():
	"""Test the steer_conv on small rotated patches"""
	N = 50
	X = np.asarray([1.,1.,1.,0.,0.,0.,-1.,-1.,-1.])
	Q = get_Q()
	X = gen_data(X, N, Q)
	X = np.reshape(X, [N,3,3,1])
	
	x = tf.placeholder('float', [None,3,3,1], name='x')
	v0 = tf.placeholder('float', [1,1,2,1], name='v0')
	v1 = tf.placeholder('float', [1,1,2,1], name='v1')
	b0 = tf.placeholder('float', [1,], name='b0')
	esc1 = equi_steer_conv(x, v0)
	esc1 = complex_relu(esc1, b0)
	z = complex_steer_conv(esc1, v1, k=1)
	
	V0 = np.random.randn(1,1,2,1)
	V1 = np.random.randn(1,1,2,1)
	B0 = np.random.rand(1)

	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		X, Y = sess.run(z, feed_dict={x : X, v0 : V0, v1 : V1, b0 : B0})
	
	R = np.sqrt(X**2 + Y**2)
	A = np.arctan2(Y, X)
	fig = plt.figure(1)
	theta = np.linspace(0, 2*np.pi, N)
	plt.plot(theta, np.squeeze(R), 'b')
	plt.plot(theta, np.squeeze(A), 'r')
	plt.show()

def small_patch_test():
	"""Test the steer_conv on small rotated patches"""
	N = 50
	X = np.random.randn(9)	#arange(9)
	Q = get_Q()
	X = gen_data(X, N, Q)
	X = np.reshape(X, [N,3,3,1])
	
	V = np.ones([3,1,1])
	#V[:,:,1,:] *= 20
	V = V/np.sqrt(np.sum(V**2))
	
	x = tf.placeholder('float', [None,3,3,1], name='x')
	v = tf.placeholder('float', [3,1,1], name='v')
	r, a = equi_real_conv(x, v)
	print V
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		X, Y = sess.run([r,a], feed_dict={x : X, v : V})
	
	R = np.sqrt(X**2 + Y**2)
	A = np.arctan2(Y, X)
	fig = plt.figure(1)
	theta = np.linspace(0, 2*np.pi, N)
	plt.plot(theta, np.squeeze(R), 'b')
	plt.plot(theta, np.squeeze(A), 'r')
	plt.show()
	

def gen_data(X, N, Q):
	# Get rotation
	theta = np.linspace(0, 2*np.pi, N)
	Y = []
	for t in theta:
		Y.append(reproject(Q,X,t))
	Y = np.vstack(Y)
	return Y

def get_Q(k=3,n=2):
	"""Return a tensor of steerable filter bases"""
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)
	x, y = np.meshgrid(lin, lin)
	gdx = gaussian_derivative(x, y, x)
	gdy = gaussian_derivative(x, y, y)
	G0 = np.reshape(gdx/np.sqrt(np.sum(gdx**2)), [k*k])
	G1 = np.reshape(gdy/np.sqrt(np.sum(gdx**2)), [k*k])
	return np.vstack([G0,G1])

def reproject(Q, X, angle):
	"""Reproject X through Q rotated by some amount"""
	# Represent in Q-space
	Y = np.dot(Q,X)
	# Rotate
	R = np.asarray([[np.cos(angle), np.sin(angle)],
					[-np.sin(angle), np.cos(angle)]])
	return np.dot(Q.T, np.dot(R,Y))
	
def dot_blade_test():
	v = tf.placeholder('float', [1,1,6,3], 'v')
	v_ = dot_blade_filter(v)

	V = np.random.randn(1,1,6,3)
	
	init = tf.initialize_all_variables()	
	with tf.Session() as sess:
		sess.run(init)
		V_ = sess.run(v_, feed_dict={v : V})

	V, V_ = V_
	print V
	print V_

def complex_batch_norm(z, phase_train, scope='bn'):
	"""bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
	batch-normalization-in-tensorflow"""
	x, y = z
	eps = 1e-4
	
	n_out = x.get_shape().as_list()[3]
	with tf.variable_scope(scope):
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma',
							trainable=True)
		
		r2 = tf.square(x) + tf.square(y)
		mean, __ = tf.nn.moments(r2, [0,1,2])
		batch_var = tf.sqrt(mean + eps)
		ema = tf.train.ExponentialMovingAverage(decay=0.99)
	
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_var)
	
		var = tf.cond(phase_train, mean_var_with_update,
					lambda: ema.average(batch_var))
		x_normed = tf.nn.batch_normalization(x, 0., var, None, gamma, 1e-3)
		y_normed = tf.nn.batch_normalization(y, 0., var, None, gamma, 1e-3)
	return x_normed, y_normed

def batch_norm(x, n_out, phase_train, scope='bn'):
    """bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
    batch-normalization-in-tensorflow"""
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta',
                           trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma',
                            trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed 


if __name__ == '__main__':
	run()
	#forward()
	#angular()
	#real_steer_comparison()
	#complex_steer_test()
	#small_patch_test()
	#complex_small_patch_test()
	#dot_blade_test()
	#Z_steer_comparison()
