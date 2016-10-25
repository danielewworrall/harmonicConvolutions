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

from steer_conv import *

from matplotlib import pyplot as plt

##### MODELS #####
	
def conv_so2(x, drop_prob, n_filters, n_classes, bs, phase_train, std_mult):
	"""The conv_so2 architecture, scatters first through an equi_real_conv
	followed by phase-pooling then summation and a nonlinearity. Current
	test time score is 92.97+/-0.06% for 3 layers deep, 15 filters"""
	# Sure layers weight & bias
	order = 3
	nf = n_filters
	
	weights = {
		'w1' : get_weights_dict([[6,],[5,],[5,]], 1, nf, std_mult=std_mult, name='W1'),
		'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W2'),
		'w3' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W3'),
		'w4' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W4'),
		'w5' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W5'),
		'w6' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W6'),
		'w7' : get_weights_dict([[6,],[5,],[5,]], nf, n_classes, std_mult=std_mult, name='W7'),
	}
	
	biases = {
		'b1' : get_bias_dict(nf, 2, name='b1'),
		'b3' : get_bias_dict(nf, 2, name='b3'),
		'b5' : get_bias_dict(nf, 2, name='b5'),
		'b7' : tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='b7')
	}
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolutional Layers
	# LAYER 1
	with tf.name_scope('block1') as scope:
		cv1 = real_input_conv(x, weights['w1'], filter_size=5, padding='SAME',
							  name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		
		# LAYER 2
		cv2 = complex_input_conv(cv1, weights['w2'], filter_size=5,
								 output_orders=[0,1,2], padding='SAME',
								 name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train)
	
	# LAYER 3
	with tf.name_scope('block3') as scope:
		cv3 = complex_input_conv(cv2, weights['w3'], filter_size=5,
								 output_orders=[0,1,2], strides=(1,2,2,1),
								 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)
		
		# LAYER 4
		cv4 = complex_input_conv(cv3, weights['w4'], filter_size=5,
								 output_orders=[0,1,2], padding='SAME',
								 name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train)
	
	# LAYER 5
	with tf.name_scope('block5') as scope:
		cv5 = complex_input_conv(cv4, weights['w5'], filter_size=5,
								 output_orders=[0,1,2], strides=(1,2,2,1),
								 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
		
		# LAYER 6
		cv6 = complex_input_conv(cv5, weights['w6'], filter_size=5,
								 output_orders=[0,1,2], padding='SAME',
								 name='6')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train)
	
	# LAYER 7
	with tf.name_scope('block7') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=5,
								 strides=(1,2,2,1), padding='SAME',
								 name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7'])

def conv_complex_bias(x, drop_prob, n_filters, n_classes, bs, phase_train, std_mult):
	"""The conv_so2 architecture, scatters first through an equi_real_conv
	followed by phase-pooling then summation and a nonlinearity. Current
	test time score is 92.97+/-0.06% for 3 layers deep, 15 filters"""
	# Sure layers weight & bias
	order = 3
	nf = n_filters
	
	weights = {
		'w1' : get_weights_dict([[6,],[5,],[5,]], 1, nf, std_mult=std_mult, name='W1'),
		'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W2'),
		'w3' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W3'),
		'w4' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W4'),
		'w7' : get_weights_dict([[6,],[5,],[5,]], nf, n_classes, std_mult=std_mult, name='W7'),
	}
	
	biases = {
		'b1' : get_bias_dict(nf, 2, name='b1'),
		'psi1' : get_bias_dict(nf, 2, rand_init=True, name='psi1'),
		'b2' : get_bias_dict(nf, 2, name='b2'),
		'psi2' : get_bias_dict(nf, 2, rand_init=True, name='psi2'),
		'b3' : get_bias_dict(nf, 2, name='b3'),
		'psi3' : get_bias_dict(nf, 2, rand_init=True, name='psi3'),
		'b4' : get_bias_dict(nf, 2, name='b4'),
		'psi4' : get_bias_dict(nf, 2, rand_init=True, name='psi4'),
		'b7' : tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='b7')
	}
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_nonlinearity(cv2, biases['b2'], tf.nn.relu)
	
	with tf.name_scope('block3') as scope:
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)

		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv4 = complex_nonlinearity(cv4, biases['b4'], tf.nn.relu)
	
	# LAYER 7
	with tf.name_scope('block7') as scope:
		cv7 = complex_input_conv(cv4, weights['w7'], filter_size=5,
								 strides=(1,2,2,1), padding='SAME',
								 name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7'])

def deep_complex_bias(x, drop_prob, n_filters, n_classes, bs, phase_train, std_mult):
	"""The conv_so2 architecture, scatters first through an equi_real_conv
	followed by phase-pooling then summation and a nonlinearity. Current
	test time score is 92.97+/-0.06% for 3 layers deep, 15 filters"""
	# Sure layers weight & bias
	order = 3
	nf = n_filters
	
	weights = {
		'w1' : get_weights_dict([[6,],[5,],[5,]], 1, nf, std_mult=std_mult, name='W1'),
		'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W2'),
		'w3' : get_weights_dict([[6,],[5,],[5,]], nf, 2*nf, std_mult=std_mult, name='W3'),
		'w4' : get_weights_dict([[6,],[5,],[5,]], 2*nf, 2*nf, std_mult=std_mult, name='W4'),
		'w5' : get_weights_dict([[6,],[5,],[5,]], 2*nf, 4*nf, std_mult=std_mult, name='W5'),
		'w6' : get_weights_dict([[6,],[5,],[5,]], 4*nf, 4*nf, std_mult=std_mult, name='W6'),
		'w7' : get_weights_dict([[6,],[5,],[5,]], 4*nf, n_classes, std_mult=std_mult, name='W7'),
	}
	
	biases = {
		'b1' : get_bias_dict(nf, 2, name='b1'),
		'psi1' : get_bias_dict(nf, 2, rand_init=True, name='psi1'),
		'b2' : get_bias_dict(nf, 2, name='b2'),
		'psi2' : get_bias_dict(nf, 2, rand_init=True, name='psi2'),
		'b3' : get_bias_dict(2*nf, 2, name='b3'),
		'psi3' : get_bias_dict(2*nf, 2, rand_init=True, name='psi3'),
		'b4' : get_bias_dict(2*nf, 2, name='b4'),
		'psi4' : get_bias_dict(2*nf, 2, rand_init=True, name='psi4'),
		'b5' : get_bias_dict(4*nf, 2, name='b5'),
		'psi5' : get_bias_dict(4*nf, 2, rand_init=True, name='psi5'),
		'b6' : get_bias_dict(4*nf, 2, name='b6'),
		'psi6' : get_bias_dict(4*nf, 2, rand_init=True, name='psi6'),
		'b7' : tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='b7')
	}
	# Reshape input picture
	x = tf.reshape(x, shape=[bs, 28, 28, 1])
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_nonlinearity(cv2, biases['b2'], tf.nn.relu)
	
	with tf.name_scope('block3') as scope:
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)

		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv4 = complex_nonlinearity(cv4, biases['b4'], tf.nn.relu)
	
	with tf.name_scope('block3') as scope:
		# LAYER 5
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)

		# LAYER 6
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv6 = complex_nonlinearity(cv6, biases['b6'], tf.nn.relu)

	# LAYER 7
	with tf.name_scope('block7') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=5,
								 strides=(1,2,2,1), padding='SAME',
								 name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7'])
	
##### CUSTOM BLOCKS FOR MODEL #####
def conv2d(X, V, b=None, strides=(1,1,1,1), padding='VALID', name='conv2d'):
    """conv2d wrapper. Supply input X, weights V and optional bias"""
    VX = tf.nn.conv2d(X, V, strides=strides, padding=padding, name=name+'_')
    if b is not None:
        VX = tf.nn.bias_add(VX, b)
    return VX

def maxpool2d(X, k=2):
    """Tied max pool. k is the stride and pool size"""
    return tf.nn.max_pool(X, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def get_weights_dict(comp_shape, in_shape, out_shape, std_mult=0.4, name='W'):
	"""Return a dict of weights for use with real_input_equi_conv. comp_shape is
	a list of the number of elements per Fourier base. For 3x3 weights use
	[3,2,2,2]. I currently assume order increasing from 0.
	"""
	weights_dict = {}
	for i, cs in enumerate(comp_shape):
		shape = cs + [in_shape,out_shape]
		weights_dict[i] = get_weights(shape, std_mult=std_mult,
									  name=name+'_'+str(i))
	return weights_dict

def get_bias_dict(n_filters, order, rand_init=False, name='b'):
	"""Return a dict of biases"""
	bias_dict = {}
	for i in xrange(order+1):
		init = 1e-2
		if rand_init:
			init = np.random.rand() * 2. *np.pi
		bias = tf.Variable(tf.constant(init, shape=[n_filters]),
						   name=name+'_'+str(i))
		bias_dict[i] = bias
	return bias_dict

def get_complex_bias_dict(n_filters, order, name='b'):
	"""Return a dict of biases"""
	bias_dict = {}
	for i in xrange(order+1):
		bias_x = tf.Variable(tf.constant(1e-2, shape=[n_filters]), name=name+'x_'+str(i))
		bias_y = tf.Variable(tf.constant(1e-2, shape=[n_filters]), name=name+'y_'+str(i))
		bias_dict[i] = (bias_x, bias_y)
	return bias_dict


##### CUSTOM FUNCTIONS FOR MAIN SCRIPT #####
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

def restore_model(saver, saveDir, sess):
	"""Save a model checkpoint"""
	save_path = saver.restore(sess, saveDir + "checkpoints/model.ckpt")
	print("Model restored from file: %s" % save_path)

def rotate_feature_maps(X, n_angles):
	"""Rotate feature maps"""
	X = np.reshape(X, [28,28])
	X_ = []
	for angle in np.linspace(0, 360, num=n_angles):
		X_.append(sciint.rotate(X, angle, reshape=False))
	X_ = np.stack(X_, axis=0)
	X_ = np.reshape(X_, [-1,784])
	return X_


##### MAIN SCRIPT #####
def run(model='conv_so2', lr=1e-2, batch_size=250, n_epochs=500, n_filters=30,
		bn_config=[False, False], trial_num='N', combine_train_val=False,
		std_mult=0.4, lr_decay=0.05):
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
	momentum=0.9
	nesterov=True
	psi_preconditioner = 5e0
	
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
	if model == 'conv_so2':
		pred = conv_so2(x, keep_prob, n_filters, n_classes, batch_size, phase_train, std_mult)
	elif model == 'conv_complex_bias':
		pred = conv_complex_bias(x, keep_prob, n_filters, n_classes, batch_size, phase_train, std_mult)
	elif model == 'deep_complex_bias':	
		pred = deep_complex_bias(x, keep_prob, n_filters, n_classes, batch_size, phase_train, std_mult)
	else:
		print('Model unrecognized')
		sys.exit(1)
	print('Using model: %s' % (model,))

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
	opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
	grads_and_vars = opt.compute_gradients(cost)
	modified_gvs = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			g = psi_preconditioner*g
		modified_gvs.append((g, v))
	optimizer = opt.apply_gradients(modified_gvs)
	
	grad_summaries_op = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			grad_summaries_op.append(tf.histogram_summary(v.name, g))
	
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
	summary = tf.train.SummaryWriter('hist_logs/', sess.graph)
	
	# Launch the graph
	sess.run(init)
	saver = tf.train.Saver()
	epoch = 0
	start = time.time()
	step = 0.
	# Keep training until reach max iterations
	while epoch < n_epochs:
		generator = minibatcher(mnist_trainx, mnist_trainy, batch_size, shuffle=True)
		cost_total = 0.
		acc_total = 0.
		vacc_total = 0.
		for i, batch in enumerate(generator):
			batch_x, batch_y = batch
			lr_current = lr/np.sqrt(1.+lr_decay*epoch)
			
			# Optimize
			feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout,
						 learning_rate : lr_current, phase_train : True}
			__, cost_, acc_, gso = sess.run([optimizer, cost, accuracy,
										grad_summaries_op], feed_dict=feed_dict)
			cost_total += cost_
			acc_total += acc_
			for summ in gso:
				summary.add_summary(summ, step)
			step += 1
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
		for summ in summaries:
			summary.add_summary(summ, step)

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
	run(model='deep_complex_bias', lr=2e-2, batch_size=80, n_epochs=500,
		std_mult=0.3, n_filters=5, combine_train_val=False)
