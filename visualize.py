'''Equivariance visualization'''

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
def deep_complex_bias(x, n_filters, n_classes, bs, std_mult):
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
	features = []
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		features.append(cv1)
		
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_nonlinearity(cv2, biases['b2'], tf.nn.relu)
		features.append(cv2)
	
	with tf.name_scope('block3') as scope:
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,1,1,1),
										 name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)
		features.append(cv3)
	'''
		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv4 = complex_nonlinearity(cv4, biases['b4'], tf.nn.relu)
		features.append(cv4)
	
	with tf.name_scope('block5') as scope:
		# LAYER 5
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', strides=(1,2,2,1),
										 name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
		features.append(cv5)

		# LAYER 6
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='4')
		cv6 = complex_nonlinearity(cv6, biases['b6'], tf.nn.relu)
		features.append(cv6)

	# LAYER 7
	with tf.name_scope('block7') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=5,
								 strides=(1,2,2,1), padding='SAME',
								 name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		cv7 = tf.nn.bias_add(cv7, biases['b7'])
		features.append(cv7)
	'''
	return features, weights, biases
	
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
	saver.restore(sess, saveDir + "checkpoints/model.ckpt")
	print("Model restored from file: %s" % saveDir + "checkpoints/model.ckpt")

def rotate_feature_maps(X, n_angles):
	"""Rotate feature maps"""
	X = np.reshape(X, [28,28])
	X_ = []
	for angle in np.linspace(0, 360, num=n_angles+1):
		X_.append(sciint.rotate(X, angle, reshape=False))
	del X_[-1]
	X_ = np.stack(X_, axis=0)
	X_ = np.reshape(X_, [-1,784])
	return X_

def rotate_90(X):
	"""Rotate feature maps by 90 deg multiples"""
	X_ = []
	for i in xrange(X.shape[0]):
		x = X[i,...]
		for j in xrange(i):
			x = np.flipud(x.T)
		X_.append(x)
	X_ = np.stack(X_, axis=0)
	return X_

def rotate_back_maps(X, n_angles):
	"""Rotate feature maps back to original orientation"""
	X_ = []
	angles = np.linspace(0, 360, num=n_angles+1)
	for i in xrange(X.shape[0]):
			X_.append(sciint.rotate(X[i,...], -angles[i], reshape=False))
	X_ = np.stack(X_, axis=0)
	return X_

def rotate_back_90(X):
	"""Rotate feature maps back to original orientation"""
	X_ = []
	for i in xrange(X.shape[0]):
		x = X[i,...]
		for j in xrange(i):
			x = np.fliplr(x.T)
		X_.append(x)
	X_ = np.stack(X_, axis=0)
	return X_

##### MAIN SCRIPT #####
def view_feature_maps(model='conv_so2', lr=1e-2, batch_size=250, n_epochs=500,
					  n_filters=30, bn_config=[False, False], trial_num='N',
					  combine_train_val=False, std_mult=0.4, lr_decay=0.05):
	tf.reset_default_graph()
	# Load dataset
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')
	mnist_trainx, mnist_trainy = mnist_train['x'], mnist_train['y']
	mnist_validx, mnist_validy = mnist_valid['x'], mnist_valid['y']
	mnist_testx, mnist_testy = mnist_test['x'], mnist_test['y']

	# Parameters
	batch_size = 1 # batch_size
	
	# Network Parameters
	n_input = 784 				# MNIST data input (img shape: 28*28)
	n_classes = 10 				# MNIST total classes (0-9 digits)
	dropout = 0.75 				# Dropout, probability to keep units
	n_filters = n_filters
	dataset_size = 10000
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	
	# Construct model
	features, __, __ = deep_complex_bias(x, n_filters, n_classes, batch_size, std_mult)
	pred = features[-1]

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			
	if combine_train_val:
		mnist_trainx = np.vstack([mnist_trainx, mnist_validx])
		mnist_trainy = np.hstack([mnist_trainy, mnist_validy])
	
	with tf.Session() as sess:
		# Launch the graph
		saver = tf.train.Saver()
		restore_model(saver, './', sess)

		# Test accuracy
		tacc_total = 0.
		test_generator = minibatcher(mnist_testx, mnist_testy, batch_size,
									 shuffle=False)
		
		input_ = mnist_trainx[np.random.randint(mnist_trainx.shape[0]),:]
		input_ = np.reshape(input_, (1,784))
		output = sess.run(features[3], feed_dict={x : input_})
		
		plt.ion()
		plt.show()
		for k, v in output.iteritems():
			print v[0].shape
			for i in xrange(v[0].shape[-1]):
				plt.subplot(1,2,1)
				plt.imshow(v[0][0,:,:,i], cmap='gray', interpolation='nearest')
				plt.subplot(1,2,2)
				plt.imshow(v[1][0,:,:,i], cmap='gray', interpolation='nearest')
				plt.draw()
				raw_input(i)


def view_filters(n_filters=5, std_mult=0.3):
	tf.reset_default_graph()	
	# Network Parameters
	n_input = 784 				# MNIST data input (img shape: 28*28)
	n_classes = 10 				# MNIST total classes (0-9 digits)
	n_filters = n_filters
	dataset_size = 10000
	batch_size = 1
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	
	# Construct model
	features, weights, biases = deep_complex_bias(x, n_filters, n_classes, batch_size, std_mult)
	
	import seaborn as sns
	sns.set_style("white")
	
	with tf.Session() as sess:
		# Launch the graph
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		
		#saver = tf.train.Saver()
		#restore_model(saver, './', sess)
				
		# Visualize weights
		plt.ion()
		plt.show()
		for i in xrange(6):
			weight_name = 'w' + str(i+1)
			bias_name = 'psi' + str(i+1)
			r = weights[weight_name]
			psi = biases[bias_name]
			q = get_complex_rotated_filters(r, psi, filter_size=5)
			Q = sess.run(q)
			print('%s: ' % (weight_name,)),
			for k, (order, pair) in enumerate(Q.iteritems()):
				for j in xrange(pair[0].shape[-1]):
					psh = pair[0].shape[-1]
					offset = k*psh
					
					R = sess.run(r[k])
					Psi = sess.run(psi[k])
					
					# Plot real part
					plt.subplot(6, 2*psh, 2*j+1+4*k*psh)
					plt.cla()
					plt.imshow(pair[0][:,:,0,j], cmap='Blues' ,interpolation='nearest')
					plt.axis('off')
					plt.subplot(6, 2*psh, 2*j+1+4*k*psh+2*psh)
					plt.cla()
					radii = [0.,1.,np.sqrt(2.),2.,np.sqrt(5.),np.sqrt(8.)]
					plt.plot(radii[(order>0):], np.squeeze(R[:,0,j]))
					plt.ylim([-1,1])
					plt.axis('off')
					
					# Plot imaginary part
					plt.subplot(6, 2*psh, 2*j+1+4*k*psh+1)
					plt.cla()
					plt.imshow(pair[1][:,:,0,j], cmap='Reds', interpolation='nearest')
					plt.axis('off')	
			plt.draw()
			raw_input()

def view_biases(n_filters=5, std_mult=0.3):
	tf.reset_default_graph()	
	# Network Parameters
	n_input = 784 				# MNIST data input (img shape: 28*28)
	n_classes = 10 				# MNIST total classes (0-9 digits)
	n_filters = n_filters
	dataset_size = 10000
	batch_size = 1
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	
	# Construct model
	features, weights, biases = deep_complex_bias(x, n_filters, n_classes, batch_size, std_mult)
	
	with tf.Session() as sess:
		# Launch the graph
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		
		saver = tf.train.Saver()
		restore_model(saver, './', sess)
				
		# Visualize weights
		plt.ion()
		plt.show()
		for i in xrange(6):
			bias_name = 'b' + str(i+1)
			b = biases[bias_name]
			B = sess.run(b)
			for k, v in B.iteritems():
				print k, np.mean(v), np.std(v)

def count_feature_maps(model='conv_so2', lr=1e-2, batch_size=250, n_epochs=500,
					  n_filters=30, bn_config=[False, False], trial_num='N',
					  combine_train_val=False, std_mult=0.4, lr_decay=0.05):
	tf.reset_default_graph()
	# Load dataset
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')
	mnist_trainx, mnist_trainy = mnist_train['x'], mnist_train['y']
	mnist_validx, mnist_validy = mnist_valid['x'], mnist_valid['y']
	mnist_testx, mnist_testy = mnist_test['x'], mnist_test['y']

	# Parameters
	batch_size = 1 # batch_size
	
	# Network Parameters
	n_input = 784 				# MNIST data input (img shape: 28*28)
	n_classes = 10 				# MNIST total classes (0-9 digits)
	dropout = 0.75 				# Dropout, probability to keep units
	n_filters = n_filters
	dataset_size = 10000
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	
	# Construct model
	features, __, __ = deep_complex_bias(x, n_filters, n_classes, batch_size, std_mult)
	pred = features[-1]

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			
	if combine_train_val:
		mnist_trainx = np.vstack([mnist_trainx, mnist_validx])
		mnist_trainy = np.hstack([mnist_trainy, mnist_validy])
	
	with tf.Session() as sess:
		# Launch the graph
		saver = tf.train.Saver()
		restore_model(saver, './', sess)

		# Test accuracy
		tacc_total = 0.
		test_generator = minibatcher(mnist_testx, mnist_testy, batch_size,
									 shuffle=False)
		
		input_ = mnist_trainx[np.random.randint(mnist_trainx.shape[0]),:]
		input_ = np.reshape(input_, (1,784))
		
		for i in xrange(6):
			output = sess.run(features[i], feed_dict={x : input_})
			print i
			for k, v in output.iteritems():
				counter = 0
				for j in xrange(v[0].shape[-1]):
					counter += np.sum(v[0][0,:,:,j]) > 0
				print('%i ' % (counter,)),
			print

def equivariance_test(model='conv_so2', lr=1e-2, batch_size=250, n_epochs=500,
					  n_filters=30, bn_config=[False, False], trial_num='N',
					  combine_train_val=False, std_mult=0.4, lr_decay=0.05):
	tf.reset_default_graph()
	# Load dataset
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')
	mnist_trainx, mnist_trainy = mnist_train['x'], mnist_train['y']
	mnist_validx, mnist_validy = mnist_valid['x'], mnist_valid['y']
	mnist_testx, mnist_testy = mnist_test['x'], mnist_test['y']

	# Parameters
	batch_size = 4
	layer = 2
	
	# Network Parameters
	n_input = 784 				# MNIST data input (img shape: 28*28)
	n_classes = 10 				# MNIST total classes (0-9 digits)
	dropout = 0.75 				# Dropout, probability to keep units
	n_filters = n_filters
	dataset_size = 10000
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	
	# Construct model
	features, __, __ = deep_complex_bias(x, n_filters, n_classes, batch_size, std_mult)	
	
	with tf.Session() as sess:
		# Launch the graph
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		
		#saver = tf.train.Saver()
		#restore_model(saver, './', sess)

		# Test accuracy
		tacc_total = 0.
		test_generator = minibatcher(mnist_testx, mnist_testy, batch_size,
									 shuffle=False)
		
		input_ = mnist_trainx[54,:]
		input_ = np.reshape(input_, (1,784))
		input_ = rotate_feature_maps(input_, batch_size)
		output = sess.run(features[layer], feed_dict={x : input_})
		
		plt.ion()
		plt.show()
		for k, v in output.iteritems():
			if k > 0:
				print v[0].shape
				for i in xrange(v[0].shape[-1]):
					# Original feature maps
					plt.figure(1)
					plt.clf()
					for j in xrange(batch_size):
						plt.subplot(2,batch_size,j+1)
						r = np.sqrt(v[0][j,:,:,i]**2 + v[1][j,:,:,i]**2)
						plt.imshow(r, cmap='jet', interpolation='nearest')
						x, y = v[0][j,:,:,i]/r, v[1][j,:,:,i]/r
						plt.quiver(x,y)
					
					# Back rotated images
					#R0 = rotate_back_maps(v[0][...,i], batch_size)
					#R1 = rotate_back_maps(v[1][...,i], batch_size)
					R0 = rotate_back_90(v[0][...,i])
					R1 = rotate_back_90(v[1][...,i])
					R = []
					for j in xrange(batch_size):
						plt.subplot(2,batch_size,batch_size+j+1)
						r = np.sqrt(R0[j,...]**2 + R1[j,...]**2)
						plt.imshow(r, cmap='jet', interpolation='nearest')
						x, y = R0[j,...]/r, R1[j,...]/r
						plt.quiver(x,y)
						R.append(r)
					R = np.stack(R, axis=0)
					deviation_map = np.std(R, axis=0)
					plt.figure(3)
					plt.imshow(deviation_map, interpolation='nearest')
					MSE = np.sum(deviation_map)
					print MSE, np.amax(deviation_map)
					plt.draw()
					raw_input(i)


if __name__ == '__main__':
	#view_feature_maps(model='deep_complex_bias', lr=2e-2, batch_size=200,
	#				  n_epochs=500, std_mult=0.3, n_filters=5, combine_train_val=False)
	#view_filters()
	#view_biases()
	#count_feature_maps(model='deep_complex_bias', lr=2e-2, batch_size=200,
	#				  n_epochs=500, std_mult=0.3, n_filters=5, combine_train_val=False)
	equivariance_test(model='deep_complex_bias', lr=2e-2, batch_size=200,
					  n_epochs=500, std_mult=0.3, n_filters=5, combine_train_val=False)



















