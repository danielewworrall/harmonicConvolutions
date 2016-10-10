'''Equivariant tests'''

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
	
def conv_so2(x, drop_prob, n_filters, n_classes, bs, phase_train, std_mult):
	"""The conv_so2 architecture, scatters first through an equi_real_conv
	followed by phase-pooling then summation and a nonlinearity. Current
	test time score is 95.12% for 3 layers deep, 15 filters"""
	# Store layers weight & bias
	order = 3
	nf = n_filters
	
	weights = {
		'w1' : get_weights_dict([[3,],[2,],[2,]], 1, nf, std_mult=std_mult, name='W1'),
		'w2' : get_weights_dict([[3,],[2,],[2,]], nf, nf, std_mult=std_mult, name='W2'),
		'w3' : get_weights_dict([[3,]], nf, nf, std_mult=std_mult, name='W3'),
		'out0' : get_weights([nf*7*7, 500], name='W4'),
		'out1': get_weights([500, n_classes], name='out')
	}
	
	biases = {
		'b1' : get_bias_dict(nf, 2, name='b1'),
		'b2' : get_bias_dict(nf, 2, name='b2'),
		'b3' : get_bias_dict(nf, 0, name='b3'),
		'out0' : tf.Variable(tf.constant(1e-2, shape=[500]), name='b4'),
		'out1': tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='out')
	}
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolutional Layers
	re1 = real_symmetric_conv(x, weights['w1'], filter_size=3, padding='SAME')
	re1 = complex_relu_dict(re1, biases['b1'])

	re2 = complex_symmetric_conv(re1, weights['w2'], filter_size=3,
								 output_orders=[0,1], strides=(1,2,2,1),
								 padding='SAME')
	re2 = complex_relu_dict(re2, biases['b2'])
	
	re3 = complex_symmetric_conv(re2, weights['w3'], filter_size=3, padding='SAME')
	re3 = complex_relu_of_sum_dict(re3, biases['b3'])[0][0]
	re4 = maxpool2d(re3, k=2)
	
	# Fully-connected layers
	fc = tf.reshape(tf.nn.dropout(re4, drop_prob), [bs, weights['out0'].get_shape().as_list()[0]])
	fc = tf.nn.bias_add(tf.matmul(fc, weights['out0']), biases['out0'])
	fc = tf.nn.relu(fc)
	fc = tf.nn.dropout(fc, drop_prob)
	
	# Output, class prediction
	return tf.nn.bias_add(tf.matmul(fc, weights['out1']), biases['out1'])

##### CUSTOM BLOCKS #####
def conv2d(X, V, b=None, strides=(1,1,1,1), padding='VALID', name='conv2d'):
    """conv2d wrapper. Supply input X, weights V and optional bias"""
    VX = tf.nn.conv2d(X, V, strides=strides, padding=padding, name=name+'_')
    if b is not None:
        VX = tf.nn.bias_add(VX, b)
    return VX

def maxpool2d(X, k=2):
    """Tied max pool. k is the stride and pool size"""
    return tf.nn.max_pool(X, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W'):
	"""Initialize weights variable with Xavier method"""
	if W_init == None:
		stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:2]))
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

def get_weights_dict(comp_shape, in_shape, out_shape, std_mult=0.4, name='W'):
	"""Return a dict of weights for use with real_symmetric_conv. comp_shape is
	a list of the number of elements per Fourier base. For 3x3 weights use
	[3,2,2,2]. I currently assume order increasing from 0.
	"""
	weights_dict = {}
	for i, cs in enumerate(comp_shape):
		shape = cs + [in_shape,out_shape]
		weights_dict[i] = get_weights(shape, std_mult=std_mult, name=name+'_'+str(i))
	return weights_dict

def get_bias_list(n_filters, order, name='b'):
	"""Return a list of biases for use with equi_real_conv()"""
	bias_list = []
	for i in xrange(order+1):
		bias = tf.Variable(tf.constant(1e-2, shape=[n_filters]), name=name+'_'+str(i))
		bias_list.append(bias)
	return bias_list

def get_bias_dict(n_filters, order, name='b'):
	"""Return a dict of biases"""
	bias_dict = {}
	for i in xrange(order+1):
		bias = tf.Variable(tf.constant(1e-2, shape=[n_filters]), name=name+'_'+str(i))
		bias_dict[i] = bias
	return bias_dict

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

def rotate_feature_maps(X, n_angles):
	"""Rotate feature maps"""
	X = np.reshape(X, [28,28])
	X_ = []
	for angle in np.linspace(0, 360, num=n_angles):
		X_.append(sciint.rotate(X, angle, reshape=False))
	X_ = np.stack(X_, axis=0)
	X_ = np.reshape(X_, [-1,784])
	return X_

def view_feature_map(n_rotations):
	"""View the rotation order m feature map"""
	tf.reset_default_graph()
	# Load dataset
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_trainx, mnist_trainy = mnist_train['x'], mnist_train['y']

	# Parameters
	batch_size = n_rotations
	
	# Network Parameters
	n_input = 784 				# MNIST data input (img shape: 28*28)
	n_classes = 10 				# MNIST total classes (0-9 digits)
	dropout = 0.75 				# Dropout, probability to keep units
	n_filters = 10
	dataset_size = 10000
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	
	pred, fm = conv_complex(x, keep_prob, n_filters, n_classes, batch_size, phase_train)
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Launch the graph
	with tf.Session() as sess:
		# Keep training until reach max iterations
		sess.run(init)
		batch_x = rotate_feature_maps(mnist_trainx[np.random.randint(10000),:], n_rotations)
		# Optimize
		feed_dict = {x: batch_x}
		fmx = sess.run(fm, feed_dict=feed_dict)
		for key, val in fmx.iteritems():
			plt.ion()
			plt.show()
			for i in xrange(n_rotations):
				plt.cla()
				plt.figure(1)
				plt.imshow(np.sqrt(val[0][i,:,:,0]**2+val[1][i,:,:,0]**2), cmap='gray', interpolation='nearest')
				plt.show()
				raw_input((key, i))
	sess.close()
		
def view_filters():
	weights = get_weights_dict([3,2,2,2], 1, 1, name='W1')
	q = get_complex_filters(weights, 3)
	
	W0 = np.reshape(np.asarray([1.000,0.368,0.135]), [3,1,1])
	W1 = np.reshape(np.asarray([0.368,0.135]), [2,1,1])
	W2 = np.reshape(np.asarray([0.368,0.135]), [2,1,1])
	W3 = np.reshape(np.asarray([0.368,0.135]), [2,1,1])
	
	with tf.Session() as sess:
		init = tf.initialize_all_variables()
		sess.run(init)
		Q = sess.run(q, feed_dict={weights[0] : W0,
								   weights[1] : W1,
								   weights[2] : W2,
								   weights[3] : W3})
		plt.ion()
		plt.show()
		for i in xrange(4):
			for j in xrange(2):
				plt.cla()
				plt.figure(1)
				print np.squeeze(Q[i][j])
				plt.imshow(np.squeeze(Q[i][j]), cmap='gray', interpolation='nearest')
				plt.draw()
				raw_input(i)

##### MAIN SCRIPT #####
def run(model='deep_steer', lr=1e-2, batch_size=250, n_epochs=500, n_filters=30,
		bn_config=[False, False], trial_num='N', combine_train_val=False, std_mult=0.4):
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
	n_input = 784 				# MNIST data input (img shape: 28*28)
	n_classes = 10 				# MNIST total classes (0-9 digits)
	dropout = 0.75 				# Dropout, probability to keep units
	n_filters = n_filters
	dataset_size = 10000
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	
	# Construct model
	if model == 'conv_complex':
		pred, __ = conv_complex(x, keep_prob, n_filters, n_classes, batch_size, phase_train)
	elif model == 'conv_so2':
		pred = conv_so2(x, keep_prob, n_filters, n_classes, batch_size, phase_train, std_mult)
	else:
		print('Model unrecognized')
		sys.exit(1)
	print('Using model: %s' % (model,))

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
	#view_feature_map(20)
	#view_filters()