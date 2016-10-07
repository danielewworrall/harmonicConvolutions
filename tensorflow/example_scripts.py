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

def conv_so2(x, drop_prob, n_filters, n_classes, bs, phase_train):
	"""The conv_so2 architecture, scatters first through an equi_real_conv
	followed by phase-pooling then summation and a nonlinearity. Current
	test time score is 95.12% for 3 layers deep, 15 filters"""
	# Store layers weight & bias
	order = 3
	nf = n_filters
	
	weights = {
		'w1' : get_weights_list([3,2,2,2], 1, nf, name='W1'),
		'w2' : get_weights_list([3,2,2,2], nf, nf, name='W2'),
		'w3' : get_weights_list([3,2,2,2], nf, nf, name='W3'),
		'w4' : get_weights_list([3,2,2,2], nf, nf, name='W4'),
		'out0' : get_weights([nf*7*7, 500], name='W4'),
		'out1': get_weights([500, n_classes], name='out')
	}
	
	biases = {
		'b1' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b1'),
		'b2' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b2'),
		'b3' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b3'),
		'b4' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b4'),
		'out0' : tf.Variable(tf.constant(1e-2, shape=[500]), name='b4'),
		'out1': tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='out')
	}
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolutional Layers
	re1 = equi_real_conv(x, weights['w1'], order=order, padding='SAME')
	re1 = tf.nn.bias_add(sum_moduli(re1), biases['b1'])
	re1 = tf.nn.relu(re1)
	
	re2 = equi_real_conv(re1, weights['w2'], order=order, padding='SAME')
	re2 = tf.nn.bias_add(sum_moduli(re2), biases['b2'])
	re2 = tf.nn.relu(re2)
	re2 = maxpool2d(re2, k=2)
	
	re3 = equi_real_conv(re2, weights['w3'], order=order, padding='SAME')
	re3 = tf.nn.bias_add(sum_moduli(re3), biases['b3'])
	re3 = tf.nn.relu(re3)
	
	re4 = equi_real_conv(re3, weights['w4'], order=order, padding='SAME')
	re4 = tf.nn.bias_add(sum_moduli(re4), biases['b4'])
	re4 = tf.nn.relu(re4)
	re4 = maxpool2d(re4, k=2)
	
	# Fully-connected layers
	fc = tf.reshape(tf.nn.dropout(re4, drop_prob), [bs, weights['out0'].get_shape().as_list()[0]])
	fc = tf.nn.bias_add(tf.matmul(fc, weights['out0']), biases['out0'])
	fc = tf.nn.relu(fc)
	fc = tf.nn.dropout(fc, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc, weights['out1']), biases['out1'])
	return out

def resnet_so2(x, drop_prob, n_filters, n_classes, bs, bn_config, phase_train):
	order = 3
	bn = bn_config
	nf = n_filters
	pt = phase_train
	# Store layers weight & bias
	weights = {
		'w0' : get_weights_list([3]+order*[2], 1, nf, name='W0'),
		'out0' : get_weights([n_filters*6*6,500], name='out0'),
		#'out1': get_weights([500, n_classes], name='out1')
		'out1': get_weights([n_filters, n_classes], name='out1')
	}
	
	biases = {
		'b0' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b0'),
		'out0': tf.Variable(tf.constant(1e-2, shape=[500])),
		'out1': tf.Variable(tf.constant(1e-2, shape=[n_classes]))
	}
	# Reshape input picture
	with tf.variable_scope('input') as scope:
		x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	with tf.variable_scope('ipconv') as scope:
		# Convolutional layer
		re0 = equi_real_conv(x, weights['w0'], order=order, padding='VALID', name=scope.name)
		re0 = tf.nn.bias_add(sum_moduli(re0), biases['b0'])
		re0 = tf.nn.relu(re0)
	
	# Residual layers
	with tf.variable_scope('residual') as scope:
		rb = residual(re0, nf, nf, order, pt, pool_in=True, bn=bn[0], name='rb1')
		rb = residual(rb, nf, nf, order, pt, pool_in=False, bn=bn[1], name='rb2')
		rb = residual(rb, nf, nf, order, pt, pool_in=False, bn=bn[2], name='rb3')

	fc = tf.reduce_mean(rb, reduction_indices=[1,2])
	
	# Output, class prediction
	with tf.variable_scope('output') as scope:
		out = tf.nn.bias_add(tf.matmul(fc, weights['out1']), biases['out1'])
		return out

def conv_complex(x, drop_prob, n_filters, n_classes, bs, phase_train):
	"""The conv_so2 architecture, with complex convolutions"""
	# Store layers weight & bias
	order = 3
	nf = n_filters
	
	weights = {
		'w1' : get_weights_list([3,2,2,2], 1, nf, name='W1'),
		'w2' : get_weights_list([3,2,2,2], nf, nf, name='W2'),
		'w2c': get_weights([3,nf,nf], name='wc2'),
		'w3' : get_weights_list([3,2,2,2], nf, nf, name='W3'),
		#'w2r' : get_weights_list([3,2,2,2], nf, nf, name='W2r'),
		#'w2c' : get_weights([3,30,10], name='W2c'),
		'out0' : get_weights([nf*7*7, 500], name='W4'),
		'out1': get_weights([500, n_classes], name='out')
	}
	
	biases = {
		'b1' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b1'),
		'b1c' : get_bias_list(nf, order=3, name='b1c'),
		'b2' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b2'),
		'b3' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b3'),
		'b4' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b4'),
		'out0' : tf.Variable(tf.constant(1e-2, shape=[500]), name='b4'),
		'out1': tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='out')
	}
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolutional Layers
	re1_ = equi_real_conv(x, weights['w1'], order=order, padding='SAME')
	# Real channel
	re1 = tf.nn.relu(tf.nn.bias_add(sum_moduli(re1_), biases['b1']))
	# Complex channel
	rc1 = complex_relu(re1_, biases['b1c'])
	rc1 = (rc1[1], rc1[2])
	
	# Real conv
	re2 = equi_real_conv(re1, weights['w2'], order=order, padding='SAME')
	# Complex conv
	rc2 = equi_complex_conv(rc1, weights['w2c'])
	
	re2 = tf.nn.relu(tf.nn.bias_add(sum_moduli(re2), biases['b2']))
	
	re3 = equi_real_conv(re2, weights['w3'], order=order, padding='SAME')
	re3 = tf.nn.relu(tf.nn.bias_add(sum_moduli(re3), biases['b3']))
	re3 = maxpool2d(re3, k=2)
	
	# Fully-connected layers
	fc = tf.reshape(tf.nn.dropout(re3, drop_prob), [bs, weights['out0'].get_shape().as_list()[0]])
	fc = tf.nn.bias_add(tf.matmul(fc, weights['out0']), biases['out0'])
	fc = tf.nn.relu(fc)
	fc = tf.nn.dropout(fc, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc, weights['out1']), biases['out1'])
	return out

def conv_nin(x, drop_prob, n_filters, n_classes, bs, phase_train):
	"""The conv_so2 architecture, scatters first through an equi_real_conv
	followed by phase-pooling then summation and a nonlinearity. Current
	test time score is 95.12% for 3 layers deep, 15 filters"""
	# Store layers weight & bias
	order = 3
	nf = n_filters
	
	weights = {
		'w1' : get_weights_list([3,2,2,2], 1, nf, name='W1'),
		'w1n' : get_weights([1,1,(order+1)*nf,nf], name='W1n'),
		'w1n2' : get_weights([1,1,nf,nf], name='W1n2'),
		'w2' : get_weights_list([3,2,2,2], nf, nf, name='W2'),
		'w2n' : get_weights([1,1,(order+1)*nf,nf], name='W2n'),
		'w2n2' : get_weights([1,1,nf,nf], name='W2n2'),
		'w3' : get_weights_list([3,2,2,2], nf, nf, name='W3'),
		'w3n' : get_weights([1,1,(order+1)*nf,nf], name='W3n'),
		'out0' : get_weights([nf*7*7, 500], name='W4'),
		'out1': get_weights([500, n_classes], name='out')
	}
	
	biases = {
		'b1' : tf.Variable(tf.constant(1e-2, shape=[4*nf]), name='b1'),
		'b1n' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b1n'),
		'b1n2' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b1n2'),
		'b2' : tf.Variable(tf.constant(1e-2, shape=[4*nf]), name='b2'),
		'b2n' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b2n'),
		'b2n2' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b2n2'),
		'b3' : tf.Variable(tf.constant(1e-2, shape=[4*nf]), name='b3'),
		'b3n' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b3n'),
		'out0' : tf.Variable(tf.constant(1e-2, shape=[500]), name='b4'),
		'out1': tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='out')
	}
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	outputs = []
	
	# Convolutional Layers
	re1 = equi_real_conv(x, weights['w1'], order=order, padding='SAME')
	re1 = tf.nn.bias_add(stack_moduli(re1), biases['b1'])
	re1 = conv2d(re1, weights['w1n'])
	re1 = tf.nn.relu(re1)
	re1 = conv2d(re1, weights['w1n2'], biases['b1n2'])
	re1 = batch_norm(re1, nf, phase_train)
	re1 = maxpool2d(tf.nn.relu(re1))
	outputs.append(re1)
	
	re2 = equi_real_conv(re1, weights['w2'], order=order, padding='SAME')
	re2 = tf.nn.bias_add(stack_moduli(re2), biases['b2'])
	re2 = conv2d(re2, weights['w2n'])
	re2 = tf.nn.relu(re2)
	re2 = conv2d(re2, weights['w2n2'], biases['b2n2'])
	re2 = batch_norm(re2, nf, phase_train)
	outputs.append(re2)
	
	re3 = equi_real_conv(re2, weights['w3'], order=order, padding='SAME')
	re3 = tf.nn.bias_add(stack_moduli(re3), biases['b3'])
	re3 = conv2d(re3, weights['w3n'])
	re3 = batch_norm(re3, nf, phase_train)
	re3 = maxpool2d(tf.nn.relu(re3))
	outputs.append(re3)
	
	# Fully-connected layers
	print re3
	fc = tf.reshape(tf.nn.dropout(re3, drop_prob), [bs, weights['out0'].get_shape().as_list()[0]])
	fc = tf.nn.bias_add(tf.matmul(fc, weights['out0']), biases['out0'])
	fc = tf.nn.relu(fc)
	fc = tf.nn.dropout(fc, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc, weights['out1']), biases['out1'])
	outputs.append(out)
	return outputs


##### CUSTOM BLOCKS #####
def residual(x, n_in, n_out, order, phase_train, pool_in=True, bn=True, name='rb'):
	W1 = get_weights_list([3]+order*[2], n_in, n_out, name=name+'W1')
	W2 = get_weights_list([3]+order*[2], n_out, n_out, name=name+'W2')
	b1 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name=name+'b1')
	#b2 = tf.Variable(tf.constant(1e-2, shape=[n_out]), name=name+'b2')
	
	with tf.variable_scope(name) as scope:
		#if pool_in:
			#x = maxpool2d(x)
		if pool_in:
			strides = (1,2,2,1)
		else:
			strides = (1,1,1,1)
		re1 = equi_real_conv(x, W1, order=order, strides=strides, padding='SAME', name=scope.name+'_re1')
		re1 = tf.nn.bias_add(sum_moduli(re1), b1)
		re1 = tf.nn.relu(re1)
			
		re2 = equi_real_conv(re1, W2, order=order, padding='SAME', name=scope.name+'_re2')
		re2 = sum_moduli(re2)

		#if bn:
		#	re2 = batch_norm(re2, n_out, phase_train, name=scope.name+'_bn')
		if pool_in:
			x = tf.nn.max_pool(x, (1,1,1,1), (1,2,2,1), padding='VALID')
		# Residual connexion---will have to adapt this later
		return 0.2*re2 + x 

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

def save_model(saver, saveDir, sess):
	"""Save a model checkpoint"""
	save_path = saver.save(sess, saveDir + "checkpoints/model.ckpt")
	print("Model saved in file: %s" % save_path)

def rotate_feature_maps(X, im_shape):
	"""Rotate feature maps"""
	Xsh = X.shape
	X = np.reshape(X, [-1,]+im_shape)
	X_ = []
	angle = []
	for i in xrange(X.shape[0]):
		angle.append(360*np.random.rand())
		X_.append(sciint.rotate(X[i,...], angle[-1], reshape=False))
	X_ = np.stack(X_, axis=0)
	X_ = np.reshape(X_, Xsh)
	angle = np.asarray(angle)
	return X_, angle

##### MAIN SCRIPT #####
def run(model='deep_steer', lr=1e-2, batch_size=250, n_epochs=500, n_filters=30,
		bn_config=[False, False], trial_num='N', combine_train_val=False):
	tf.reset_default_graph()
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
		# A standard Z-convolution network
		pred = conv_Z(x, keep_prob, n_filters, n_classes)
	elif model == 'conv_so2':
		# Rotational pooling network [SO(2)-convolution]: conv-pool-sum-relu
		pred = conv_so2(x, keep_prob, n_filters, n_classes, batch_size, phase_train)
	elif model == 'resnet_so2':
		# Experimentation with resnets and SO(2)-convolution
		pred= resnet_so2(x, keep_prob, n_filters, n_classes, batch_size, bn_config, phase_train)
	elif model == 'conv_nin':
		pred = conv_nin(x, keep_prob, n_filters, n_classes, batch_size, phase_train)
	elif model == 'conv_complex':
		pred = conv_complex(x, keep_prob, n_filters, n_classes, batch_size, phase_train)
	else:
		print('Model unrecognized')
		sys.exit(1)
	print('Using model: %s' % (model,))
	#pred = pred[-1]

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
	
	# Summary writers
	acc_ph = tf.placeholder(tf.float32, [], name='acc_')
	acc_op = tf.scalar_summary("Validation Accuracy", acc_ph)
	cost_ph = tf.placeholder(tf.float32, [], name='cost_')
	cost_op = tf.scalar_summary("Training Cost", cost_ph)
	lr_ph = tf.placeholder(tf.float32, [], name='lr_')
	lr_op = tf.scalar_summary("Learning Rate", lr_ph)
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
	summary = tf.train.SummaryWriter('logs/', sess.graph)
	
	# Launch the graph
	sess.run(init)
	saver = tf.train.Saver()
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
			#lr_current = lr/(10.**np.floor(epoch/150))
			
			rot_x, angles_ = rotate_feature_maps(batch_x, [28,28])
			
			# Optimize
			feed_dict = {x: rot_x, y: batch_y, keep_prob: dropout,
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
		
		feed_dict={cost_ph : cost_total, acc_ph : vacc_total, lr_ph : lr_current}
		summaries = sess.run([cost_op, acc_op, lr_op], feed_dict=feed_dict)
		summary.add_summary(summaries[0], epoch)
		summary.add_summary(summaries[1], epoch)
		summary.add_summary(summaries[2], epoch)

		print "[" + str(trial_num),str(epoch) + \
			"], Minibatch Loss: " + \
			"{:.6f}".format(cost_total) + ", Train Acc: " + \
			"{:.5f}".format(acc_total) + ", Time: " + \
			"{:.5f}".format(time.time()-start) + ", Val acc: " + \
			"{:.5f}".format(vacc_total)
		epoch += 1
		
		if (epoch) % 50 == 0:
			save_model(saver, './', sess)
	
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
	save_model(saver, './', sess)
	sess.close()
	return tacc_total



if __name__ == '__main__':
	run(model='conv_so2', lr=1e-3, batch_size=132, n_epochs=500,
		n_filters=10, combine_train_val=False, bn_config=[True,True,True])
