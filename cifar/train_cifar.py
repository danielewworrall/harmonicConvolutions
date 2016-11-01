'''Equivariant tests'''

import os
import sys
import time
sys.path.append('../')

import cv2
import numpy as np
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import tensorflow as tf

from cifar10 import get_cifar10_data
from steer_conv import *

from matplotlib import pyplot as plt

def all_cnn(x, n_filters, n_classes, bs, phase_train, std_mult, filter_gain):
	"""The deep_complex_bias architecture adapted for CIFAR"""
	# Sure layers weight & bias
	nf = 3*n_filters
	nf2 = 3*int(n_filters*filter_gain)
	
	weights = {
		'w1' : get_weights([3,3,3,nf], std_mult=1., name='W1'),
		'w2' : get_weights([3,3,nf,nf], std_mult=1., name='W2'),
		'w3' : get_weights([3,3,nf,nf], std_mult=1., name='W3'),
		'w4' : get_weights([3,3,nf,nf2], std_mult=1., name='W4'),
		'w5' : get_weights([3,3,nf2,nf2], std_mult=1., name='W5'),
		'w6' : get_weights([3,3,nf2,nf2], std_mult=1., name='W6'),
		'w7' : get_weights([3,3,nf2,nf2], std_mult=1., name='W7'),
		'w8' : get_weights([1,1,nf2,nf2], std_mult=1., name='W8'),
		'w9' : get_weights([1,1,nf2,n_classes], std_mult=1., name='W9')
	}
	
	biases = {
		'b1' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b1'),
		'b2' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b2'),
		'b3' : tf.Variable(tf.constant(1e-2, shape=[nf]), name='b3'),
		'b4' : tf.Variable(tf.constant(1e-2, shape=[nf2]), name='b4'),
		'b5' : tf.Variable(tf.constant(1e-2, shape=[nf2]), name='b5'),
		'b6' : tf.Variable(tf.constant(1e-2, shape=[nf2]), name='b6'),
		'b7' : tf.Variable(tf.constant(1e-2, shape=[nf2]), name='b8'),
		'b8' : tf.Variable(tf.constant(1e-2, shape=[nf2]), name='b8'),
		'b9' : tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='b9')}

	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = conv2d(x, weights['w1'], b=biases['b1'], padding='SAME', name='1')
		cv2 = conv2d(tf.nn.relu(cv1), weights['w2'], b=biases['b2'], padding='SAME', name='2')
		cv3 = conv2d(tf.nn.relu(cv2), weights['w3'], b=biases['b3'], strides=(1,2,2,1), padding='SAME', name='3')
	
	with tf.name_scope('block2') as scope:
		cv4 = conv2d(tf.nn.relu(cv3), weights['w4'], b=biases['b4'], padding='SAME', name='4')
		cv5 = conv2d(tf.nn.relu(cv4), weights['w5'], b=biases['b5'], padding='SAME', name='5')
		cv6 = conv2d(tf.nn.relu(cv5), weights['w6'], b=biases['b6'], strides=(1,2,2,1), padding='SAME', name='6')
	
	with tf.name_scope('block3') as scope:
		cv7 = conv2d(tf.nn.relu(cv6), weights['w7'], b=biases['b7'], padding='SAME', name='7')		
		cv8 = conv2d(tf.nn.relu(cv7), weights['w8'], b=biases['b8'], name='8')
		cv9 = conv2d(tf.nn.relu(cv8), weights['w9'], b=biases['b9'], name='9')
		print cv9
		return tf.reduce_mean(cv9, reduction_indices=[1,2])

def steer_net(x, n_filters, n_classes, bs, phase_train, std_mult, filter_gain):
	"""The deep_complex_bias architecture adapted for CIFAR"""
	# Sure layers weight & bias
	order = 3
	nf = n_filters
	nf2 = int(n_filters*filter_gain)
	
	weights = {
		'w1' : get_weights_dict([[6,],[5,],[5,]], 3, nf, std_mult=std_mult, name='W1'),
		'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W2'),
		'w3' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W3'),
		'w4' : get_weights_dict([[3,],[2,],[2,]], nf, nf2, std_mult=std_mult, name='W4'),
		'w5' : get_weights_dict([[3,],[2,],[2,]], nf2, nf2, std_mult=std_mult, name='W5'),
		'w6' : get_weights_dict([[3,],[2,],[2,]], nf2, nf2, std_mult=std_mult, name='W6'),
		'w7' : get_weights_dict([[3,],[2,],[2,]], nf2, nf2, std_mult=std_mult, name='W7'),
		'w8' : get_weights([1,1,nf2,nf2], std_mult=1., name='W8'),
		'w9' : get_weights([1,1,nf2,n_classes], std_mult=1., name='W9')
	}
	
	biases = {
		'b1' : get_bias_dict(nf, 2, name='b1'),
		'b2' : get_bias_dict(nf, 2, name='b2'),
		'b3' : get_bias_dict(nf, 2, name='b3'),
		'b4' : get_bias_dict(nf2, 2, name='b4'),
		'b5' : get_bias_dict(nf2, 2, name='b5'),
		'b6' : get_bias_dict(nf2, 2, name='b6'),
		'b7' : get_bias_dict(nf2, 2, name='b7'),
		'b8' : tf.Variable(tf.constant(1e-2, shape=[nf2]), name='b8'),
		'b9' : tf.Variable(tf.constant(1e-2, shape=[n_classes]), name='b9')}
	
	psis = {
		'p1' : get_phase_dict(1, nf, 2, name='p1'),
		'p2' : get_phase_dict(nf, nf, 2, name='p2'),
		'p3' : get_phase_dict(nf, nf, 2, name='p3'),
		'p4' : get_phase_dict(nf, nf2, 2, name='p4'),
		'p5' : get_phase_dict(nf2, nf2, 2, name='p5'),
		'p6' : get_phase_dict(nf2, nf2, 2, name='p6'),
		'p7' : get_phase_dict(nf2, nf2, 2, name='p7')
	}
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], psis['p1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], psis['p2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train)
		# LAYER 3
		cv2 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], psis['p3'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)
	
	with tf.name_scope('block2') as scope:
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], psis['p4'],
									  filter_size=3, padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train)
		# LAYER 2
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], psis['p5'],
										 filter_size=3, output_orders=[0,1,2],
										 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
		# LAYER 3
		cv5 = mean_pooling(cv5, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], psis['p6'],
										 filter_size=3, output_orders=[0,1,2],
										 padding='SAME', name='6')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train)
	
	with tf.name_scope('block3') as scope:
		cv7 = complex_input_rotated_conv(cv6, weights['w7'], psis['p7'],
									  filter_size=3, padding='SAME', name='7')
		cv7 = complex_nonlinearity(cv7, biases['b7'], tf.nn.relu)
		cv7 = stack_magnitudes(cv7)
		
		cv8 = conv2d(cv7, weights['w8'], b=biases['b8'], name='8')
		cv9 = conv2d(tf.nn.relu(cv8), weights['w9'], b=biases['b9'], name='9')
		return tf.reduce_mean(cv9, reduction_indices=[1,2])

	
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

def get_bias_dict(n_filters, order, name='b'):
	"""Return a dict of biases"""
	bias_dict = {}
	for i in xrange(order+1):
		bias = tf.Variable(tf.constant(1e-2, shape=[n_filters]),
						   name=name+'_'+str(i))
		bias_dict[i] = bias
	return bias_dict

def get_phase_dict(n_in, n_out, order, name='b'):
	"""Return a dict of phase offsets"""
	phase_dict = {}
	for i in xrange(order+1):
		init = np.random.rand(1,1,n_in,n_out) * 2. *np.pi
		init = np.float32(init)
		phase = tf.Variable(tf.constant(init, shape=[1,1,n_in,n_out]),
						   name=name+'_'+str(i))
		phase_dict[i] = phase
	return phase_dict


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

def save_model(saver, saveDir, sess, saveSubDir=''):
	"""Save a model checkpoint"""
	dir_ = saveDir + "checkpoints/" + saveSubDir
	if not os.path.exists(dir_):
		os.mkdir(dir_)
		print("Created: %s" % (dir_))
	save_path = saver.save(sess, dir_ + "/model.ckpt")
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

def get_learning_rate(current, best, counter, learning_rate, delay=15):
    """If have not seen accuracy improvement in delay epochs, then divide 
    learning rate by 10
    """
    if current > best:
        best = current
        counter = 0
    elif counter > delay:
        learning_rate = learning_rate / 10.
        counter = 0
    else:
        counter += 1
    return (best, counter, learning_rate)

##### MAIN SCRIPT #####
def run(opt):
	tf.reset_default_graph()
	# Load dataset
	train_x, train_y, test_x, test_y = get_cifar10_data('../data/cifar10',
														'train_all.npz',
														'val.npz')
	train_x = np.transpose(train_x, (0,2,3,1))
	test_x = np.transpose(test_x, (0,2,3,1))
	
	valid_x = train_x[40000:,...]
	valid_y = train_y[40000:]
	train_x = train_x[:40000,...]
	train_y = train_y[:40000]
	
	# Parameters
	nesterov=True
	model = opt['model']
	lr = opt['lr']
	batch_size = opt['batch_size']
	n_epochs = opt['n_epochs']
	n_filters = opt['n_filters']
	trial_num = opt['trial_num']
	combine_train_val = opt['combine_train_val']
	std_mult = opt['std_mult']
	filter_gain = opt['filter_gain']
	momentum = opt['momentum']
	psi_preconditioner = opt['psi_preconditioner']
	delay = opt['delay']
	model_dir = 'cifar/trial'+str(trial_num)
	display_step=50
	save_step = 5
	
	# Network Parameters
	n_classes = 10
	dataset_size = train_x.shape[0]
	# tf Graph input
	tsh = train_x.shape
	x = tf.placeholder(tf.float32, [batch_size,tsh[1],tsh[2],tsh[3]])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	
	# Construct model
	if model=='all_cnn':
		pred = all_cnn(x, n_filters, n_classes, batch_size, phase_train, std_mult, filter_gain)
	elif model=='steer_net':
		pred = steer_net(x, n_filters, n_classes, batch_size, phase_train, std_mult, filter_gain)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
	opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,
									 momentum=momentum, use_nesterov=nesterov)
	#opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
	print('  Constructed loss')
	
	grads_and_vars = opt.compute_gradients(cost)
	modified_gvs = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			g = psi_preconditioner*g
		modified_gvs.append((g, v))
	optimizer = opt.apply_gradients(modified_gvs)
	print('  Optimizer built')
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	print('  Evaluation metric constructed')
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	print('  Variables initialized')
	
	if combine_train_val:
		train_x = np.vstack([train_x, valid_x])
		train_y = np.hstack([train_y, valid_y])
	
	# Summary writers
	acc_ph = tf.placeholder(tf.float32, [], name='acc_')
	acc_op = tf.scalar_summary("Validation Accuracy", acc_ph)
	cost_ph = tf.placeholder(tf.float32, [], name='cost_')
	cost_op = tf.scalar_summary("Training Cost", cost_ph)
	lr_ph = tf.placeholder(tf.float32, [], name='lr_')
	lr_op = tf.scalar_summary("Learning Rate", lr_ph)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = False
	config.inter_op_parallelism_threads = 1 #prevent inter-session threads?
	sess = tf.Session(config=config)
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
	summary_dir = '../logs/cifar/trial'+str(trial_num)
	if not os.path.exists(summary_dir):
		os.mkdir(summary_dir)
		print("Created: %s" % (summary_dir))
	summary = tf.train.SummaryWriter(summary_dir, sess.graph)
	print('  Summaries constructed')
	
	# Launch the graph
	sess.run(init)
	saver = tf.train.Saver()
	epoch = 0
	start = time.time()
	step = 0.
	lr_current = lr
	counter = 0
	best = 0.
	print('  Begin training')
	# Keep training until reach max iterations
	while epoch < n_epochs:
		generator = minibatcher(train_x, train_y, batch_size, shuffle=True)
		cost_total = 0.
		acc_total = 0.
		vacc_total = 0.
		for i, batch in enumerate(generator):
			batch_x, batch_y = batch

			# Optimize
			feed_dict = {x: batch_x, y: batch_y, learning_rate : lr_current,
						 phase_train : True}
			__, cost_, acc_ = sess.run([optimizer, cost, accuracy],
				feed_dict=feed_dict)
			if np.isnan(cost_):
				print
				print('Oops: Training went unstable')
				print
				return -1
			if step % display_step == 0:
				print('  Training accuracy: %f' % (acc_,))
			cost_total += cost_
			acc_total += acc_
			step += 1
		cost_total /=(i+1.)
		acc_total /=(i+1.)
		
		if not combine_train_val:
			val_generator = minibatcher(valid_x, valid_y, batch_size, shuffle=False)
			for i, batch in enumerate(val_generator):
				batch_x, batch_y = batch
				# Calculate batch loss and accuracy
				feed_dict = {x: batch_x, y: batch_y, phase_train : False}
				vacc_ = sess.run(accuracy, feed_dict=feed_dict)
				vacc_total += vacc_
			vacc_total = vacc_total/(i+1.)
		
		feed_dict={cost_ph : cost_total, acc_ph : vacc_total, lr_ph : lr_current}
		summaries = sess.run([cost_op, acc_op, lr_op], feed_dict=feed_dict)
		for summ in summaries:
			summary.add_summary(summ, step)

		best, counter, lr_current = get_learning_rate(vacc_total, best, counter,
													  lr_current, delay=delay)
		
		print "[" + str(trial_num),str(epoch) + \
			"], Minibatch Loss: " + \
			"{:.6f}".format(cost_total) + ", Train Acc: " + \
			"{:.5f}".format(acc_total) + ", Time: " + \
			"{:.5f}".format(time.time()-start) + ", Counter: " + \
			"{:2d}".format(counter) + ", Val acc: " + \
			"{:.5f}".format(vacc_total)
		epoch += 1
				
		if (epoch) % save_step == 0:
			save_model(saver, '../', sess, saveSubDir=model_dir)
	
	print "Testing"
	
	# Test accuracy
	tacc_total = 0.
	test_generator = minibatcher(test_x, test_y, batch_size, shuffle=False)
	for i, batch in enumerate(test_generator):
		batch_x, batch_y = batch
		feed_dict={x: batch_x, y: batch_y, phase_train : False}
		tacc = sess.run(accuracy, feed_dict=feed_dict)
		tacc_total += tacc
	tacc_total = tacc_total/(i+1.)
	print('Test accuracy: %f' % (tacc_total,))
	save_model(saver, '../', sess, saveSubDir=model_dir)
	sess.close()
	return tacc_total

if __name__ == '__main__':
	opt = {}
	opt['model'] = 'all_cnn'
	opt['lr'] = 1e-2
	opt['batch_size'] = 64
	opt['n_epochs'] = 50
	opt['n_filters'] = 32
	opt['trial_num'] = 'M'
	opt['combine_train_val'] = False
	opt['std_mult'] = 1.
	opt['filter_gain'] = 2
	opt['momentum'] = 0.93
	opt['psi_preconditioner'] = 1.
	opt['delay'] = 13
	run(opt)
