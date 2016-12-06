"""
Harmonic Network Definitions

This file contains the definition of our neural network models. This
should be used an example of how to create a harmonic network using our 
helper functions in tensorflow.

"""
import tensorflow as tf

from harmonic_network_helpers import *
from harmonic_network_components import *

def deep_Z(opt, x, phase_train, device='/cpu:0'):
	"""A standard neural net"""
	# Sure layers weight & bias
	nf = opt['n_filters']
	bs = opt['batch_size']

	with tf.device(device):
		weights = {
			'w1' : get_weights([3,3,1,nf], std_mult=1., name='W1', device=device),
			'w2' : get_weights([3,3,nf,nf], std_mult=1., name='W2', device=device),
			'w3' : get_weights([3,3,nf,nf], std_mult=1., name='W3', device=device),
			'w4' : get_weights([3,3,nf,nf], std_mult=1., name='W4', device=device),
			'w5' : get_weights([3,3,nf,nf], std_mult=1., name='W5', device=device),
			'w6' : get_weights([3,3,nf,nf], std_mult=1., name='W6', device=device),
			'w7' : get_weights([4,4,nf,opt['n_classes']], std_mult=1., name='W7', device=device)
		}
		
		biases = {
			'b1' : tf.get_variable('b1', dtype=tf.float32, shape=[nf],
								   initializer=tf.constant_initializer(1e-2)),
			'b2' : tf.get_variable('b2', dtype=tf.float32, shape=[nf],
								   initializer=tf.constant_initializer(1e-2)),
			'b3' : tf.get_variable('b3', dtype=tf.float32, shape=[nf],
								   initializer=tf.constant_initializer(1e-2)),
			'b4' : tf.get_variable('b4', dtype=tf.float32, shape=[nf],
								   initializer=tf.constant_initializer(1e-2)),
			'b5' : tf.get_variable('b5', dtype=tf.float32, shape=[nf],
								   initializer=tf.constant_initializer(1e-2)),
			'b6' : tf.get_variable('b6', dtype=tf.float32, shape=[nf],
								   initializer=tf.constant_initializer(1e-2)),
			'b7' : tf.get_variable('b7', dtype=tf.float32, shape=[opt['n_classes']],
								   initializer=tf.constant_initializer(1e-2)),
		}
		# Reshape input picture -- square inputs for now
		size = opt['dim'] - 2*opt['crop_shape']
		x = tf.reshape(x, shape=[bs,size,size,opt['n_channels']])
	
	fms = []
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = conv2d(x, weights['w1'], biases['b1'], name='cv1')
		cv2 = conv2d(tf.nn.relu(cv1), weights['w2'], biases['b2'], name='cv2')
		cv2 = batch_norm(cv2, phase_train, name='bn2')
		cv2 = maxpool2d(cv2, k=2)
		cv3 = conv2d(tf.nn.relu(cv2), weights['w3'], biases['b3'], name='cv3')
		cv4 = conv2d(tf.nn.relu(cv3), weights['w4'], biases['b4'], name='cv4')
		cv4 = batch_norm(cv4, phase_train, name='bn4')
		cv5 = conv2d(tf.nn.relu(cv4), weights['w5'], biases['b5'], name='cv5')
		cv6 = conv2d(tf.nn.relu(cv5), weights['w6'], biases['b6'], name='cv6')
		cv6 = batch_norm(cv6, phase_train, name='bn6')
		cv7 = conv2d(tf.nn.relu(cv6), weights['w7'], biases['b7'], name='cv7')
		return tf.squeeze(cv7)

def deep_stable(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = opt['n_filters']
	nf2 = int(nf*opt['filter_gain'])
	nf3 = int(nf*(opt['filter_gain']**2.))
	bs = opt['batch_size']
	n = ((opt['filter_size']+1)/2)
	tr1 = (n*(n+1))/2
	tr2 = tr1 - 1
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1' : get_weights_dict([[tr1,],[tr2,]], opt['n_channels'], nf, std_mult=sm, name='W1', device=device),
			'w2' : get_weights_dict([[tr1,],[tr2,]], nf, nf, std_mult=sm, name='W2', device=device),
			'w3' : get_weights_dict([[tr1,],[tr2,]], nf, nf2, std_mult=sm, name='W3', device=device),
			'w4' : get_weights_dict([[tr1,],[tr2,]], nf2, nf2, std_mult=sm, name='W4', device=device),
			'w5' : get_weights_dict([[tr1,],[tr2,]], nf2, nf3, std_mult=sm, name='W5', device=device),
			'w6' : get_weights_dict([[tr1,],[tr2,]], nf3, nf3, std_mult=sm, name='W6', device=device),
			'w7' : get_weights_dict([[tr1,],[tr2,]], nf3, opt['n_classes'], std_mult=sm, name='W7', device=device),
		}
		
		biases = {
			'b1' : get_bias_dict(nf, order, name='b1', device=device),
			'b2' : get_bias_dict(nf, order, name='b2', device=device),
			'b3' : get_bias_dict(nf2, order, name='b3', device=device),
			'b4' : get_bias_dict(nf2, order, name='b4', device=device),
			'b5' : get_bias_dict(nf3, order, name='b5', device=device),
			'b6' : get_bias_dict(nf3, order, name='b6', device=device),
			'b7' : tf.get_variable('b7', dtype=tf.float32, shape=[opt['n_classes']],
				initializer=tf.constant_initializer(1e-2)),
			'psi1' : get_phase_dict(1, nf, order, name='psi1', device=device),
			'psi2' : get_phase_dict(nf, nf, order, name='psi2', device=device),
			'psi3' : get_phase_dict(nf, nf2, order, name='psi3', device=device),
			'psi4' : get_phase_dict(nf2, nf2, order, name='psi4', device=device),
			'psi5' : get_phase_dict(nf2, nf3, order, name='psi5', device=device),
			'psi6' : get_phase_dict(nf3, nf3, order, name='psi6', device=device)
		}
		# Reshape input picture -- square inputs for now
		size = opt['dim'] - 2*opt['crop_shape']
		x = tf.reshape(x, shape=[bs,size,size,opt['n_channels']])
	
	fms = []
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=opt['filter_size'], padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		fms.append(cv1)	
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=opt['filter_size'], output_orders=[0,1],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train,
								 name='batchNorm1', device=device)
		fms.append(cv2)
	with tf.name_scope('block2') as scope:
		cv2 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
										 filter_size=opt['filter_size'], output_orders=[0,1],
										 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)
		fms.append(cv3)
		# LAYER 4
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=opt['filter_size'], output_orders=[0,1],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train,
								 name='batchNorm2', device=device)
		fms.append(cv4)
	with tf.name_scope('block3') as scope:
		cv4 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 5
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=opt['filter_size'], output_orders=[0,1],
										 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
		fms.append(cv5)
		# LAYER 6
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=opt['filter_size'], output_orders=[0,1],
										 padding='SAME', name='4')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train,
								 name='batchNorm3', device=device)
		fms.append(cv6)
	# LAYER 7
	with tf.name_scope('block4') as scope:
		cv7 = complex_input_conv(cv6, weights['w7'], filter_size=opt['filter_size'],
								 padding='SAME', name='7')
		cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
		return tf.nn.bias_add(cv7, biases['b7']) #, fms

def deep_plankton(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = opt['n_filters']
	nf2 = int(nf*opt['filter_gain'])
	nf3 = int(nf*(opt['filter_gain']**2.))
	bs = opt['batch_size']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1' : get_weights_dict([[6,],[5,],[5,]], opt['n_channels'], nf, std_mult=sm, name='W1', device=device),
			'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=sm, name='W2', device=device),
			'w3' : get_weights_dict([[6,],[5,]], nf, nf2, std_mult=sm, name='W3', device=device),
			'w4' : get_weights_dict([[6,],[5,]], nf2, nf2, std_mult=sm, name='W4', device=device),
			'w5' : get_weights_dict([[6,],[5,]], nf2, nf3, std_mult=sm, name='W5', device=device),
			'w6' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W6', device=device),
			'w7' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W7', device=device),
			'w8' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W8', device=device),
			'w9' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W9', device=device),
			'w10' : get_weights_dict([[6,],[5,]], nf3, nf3, std_mult=sm, name='W10', device=device),
			'w11' : get_weights_dict([[6,],[5,]], nf3, opt['n_classes'], std_mult=sm, name='W11', device=device),
		}
		
		biases = {
			'b11' : tf.get_variable('b11', dtype=tf.float32, shape=[opt['n_classes']],
				initializer=tf.constant_initializer(1e-2)),
			'psi1' : get_phase_dict(1, nf, order+1, name='psi1', device=device),
			'psi2' : get_phase_dict(nf, nf, order+1, name='psi2', device=device),
			'psi3' : get_phase_dict(nf, nf2, order, name='psi3', device=device),
			'psi4' : get_phase_dict(nf2, nf2, order, name='psi4', device=device),
			'psi5' : get_phase_dict(nf2, nf3, order, name='psi5', device=device),
			'psi6' : get_phase_dict(nf3, nf3, order, name='psi6', device=device),
			'psi7' : get_phase_dict(nf3, nf3, order, name='psi7', device=device),
			'psi8' : get_phase_dict(nf3, nf3, order, name='psi8', device=device),
			'psi9' : get_phase_dict(nf3, nf3, order, name='psi9', device=device),
			'psi10' : get_phase_dict(nf3, nf3, order, name='psi10', device=device)
		}
		# Reshape input picture -- square inputs for now
		size = opt['dim'] - 2*opt['crop_shape']
		x = tf.reshape(x, shape=[bs,size,size,opt['n_channels']])
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_batch_norm(cv1, tf.nn.relu, phase_train, name='bn1', device=device)
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train, name='bn2', device=device)
	
	with tf.name_scope('block2') as scope:
		cv3 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		# LAYER 3
		cv3 = complex_input_rotated_conv(cv3, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='3')
		cv3 = complex_batch_norm(cv3, tf.nn.relu, phase_train, name='bn3', device=device)
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train, name='b4', device=device)
	
	with tf.name_scope('block3') as scope:
		cv5 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv5 = complex_input_rotated_conv(cv5, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='5')
		cv5 = complex_batch_norm(cv5, tf.nn.relu, phase_train, name='bn5', device=device)
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='6')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train, name='bn6', device=device)
		cv7 = complex_input_rotated_conv(cv6, weights['w7'], biases['psi7'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='7')
		cv7 = complex_batch_norm(cv7, tf.nn.relu, phase_train, name='bn7', device=device)
	
	with tf.name_scope('block4') as scope:
		cv8 = mean_pooling(cv7, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv8 = complex_input_rotated_conv(cv8, weights['w8'], biases['psi8'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='8')
		cv8 = complex_batch_norm(cv8, tf.nn.relu, phase_train, name='bn8', device=device)
		cv9 = complex_input_rotated_conv(cv8, weights['w9'], biases['psi9'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='9')
		cv9 = complex_batch_norm(cv9, tf.nn.relu, phase_train, name='bn9', device=device)
		cv10 = complex_input_rotated_conv(cv9, weights['w10'], biases['psi10'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='10')
		cv10 = complex_batch_norm(cv10, tf.nn.relu, phase_train, name='bn10', device=device)
	
	with tf.name_scope('block5') as scope:
		cv11 = complex_input_conv(cv10, weights['w11'], filter_size=5,
								 padding='SAME', name='11')
		cv11 = tf.reduce_mean(sum_magnitudes(cv11), reduction_indices=[1,2])
		return tf.nn.bias_add(cv11, biases['b11'])

def deep_cifar(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = opt['n_filters']
	nf2 = int(nf*opt['filter_gain'])
	nf3 = int(nf*(opt['filter_gain']**2.))
	bs = opt['batch_size']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1' : get_weights_dict([[6,],[5,],[5,]], opt['n_channels'], nf, std_mult=sm, name='W1', device=device),
			'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=sm, name='W2', device=device),
			'w3' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=sm, name='W3', device=device),
			'w4' : get_weights_dict([[6,],[5,]], nf, nf2, std_mult=sm, name='W4', device=device),
			'w5' : get_weights_dict([[6,],[5,]], nf2, nf2, std_mult=sm, name='W5', device=device),
			'w6' : get_weights_dict([[6,],[5,]], nf2, nf2, std_mult=sm, name='W6', device=device),
			'w7' : get_weights_dict([[6,],[5,]], nf2, nf3, std_mult=sm, name='W7', device=device),
			'w8' : get_weights_dict([[3,],[2,]], nf3, nf3, std_mult=sm, name='W8', device=device),
			'w9' : get_weights_dict([[3,],[2,]], nf3, opt['n_classes'], std_mult=sm, name='W9', device=device),
		}
		
		biases = {
			'b1' : get_bias_dict(nf, order+1, name='b1', device=device),
			'b2' : get_bias_dict(nf, order+1, name='b2', device=device),
			'b3' : get_bias_dict(nf, order, name='b3', device=device),
			'b4' : get_bias_dict(nf2, order, name='b4', device=device),
			'b5' : get_bias_dict(nf2, order, name='b5', device=device),
			'b6' : get_bias_dict(nf2, order, name='b6', device=device),
			'b7' : get_bias_dict(nf3, order, name='b7', device=device),
			'b8' : get_bias_dict(nf3, order, name='b8', device=device),
			'b9' : tf.get_variable('b9', dtype=tf.float32, shape=[opt['n_classes']],
				initializer=tf.constant_initializer(1e-2)),
			'psi1' : get_phase_dict(1, nf, order+1, name='psi1', device=device),
			'psi2' : get_phase_dict(nf, nf, order+1, name='psi2', device=device),
			'psi3' : get_phase_dict(nf, nf, order+1, name='psi3', device=device),
			'psi4' : get_phase_dict(nf, nf2, order, name='psi4', device=device),
			'psi5' : get_phase_dict(nf2, nf2, order, name='psi5', device=device),
			'psi6' : get_phase_dict(nf2, nf2, order, name='psi6', device=device),
			'psi7' : get_phase_dict(nf2, nf3, order, name='psi7', device=device),
			'psi8' : get_phase_dict(nf3, nf3, order, name='psi8', device=device),
		}
		# Reshape input picture -- square inputs for now
		size = opt['dim'] - 2*opt['crop_shape']
		x = tf.reshape(x, shape=[bs,size,size,opt['n_channels']])
	
	# Convolutional Layers
	with tf.name_scope('block1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
										 filter_size=5, output_orders=[0,1,2],
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train, name='bn1', device=device)	
		cv3 = mean_pooling(cv2, ksize=(1,3,3,1), strides=(1,2,2,1))
		cv3 = complex_input_rotated_conv(cv3, weights['w3'], biases['psi3'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='3')
		cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)

	with tf.name_scope('block2') as scope:
		cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='4')
		cv4 = complex_batch_norm(cv4, tf.nn.relu, phase_train, name='bn2', device=device)
		cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='5')
		cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
		cv5 = mean_pooling(cv5, ksize=(1,3,3,1), strides=(1,2,2,1))
		cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='6')
		cv6 = complex_batch_norm(cv6, tf.nn.relu, phase_train, name='bn3', device=device)

	with tf.name_scope('block3') as scope:
		cv7 = complex_input_rotated_conv(cv6, weights['w7'], biases['psi7'],
										 filter_size=5, output_orders=[0,1],
										 padding='SAME', name='7')
		cv7 = complex_nonlinearity(cv7, biases['b7'], tf.nn.relu)
		cv8 = complex_input_rotated_conv(cv7, weights['w8'], biases['psi8'],
										 filter_size=3, output_orders=[0,1],
										 padding='SAME', name='8')
		cv8 = complex_batch_norm(cv8, tf.nn.relu, phase_train, name='bn4', device=device)
		cv9 = complex_input_conv(cv8, weights['w9'], filter_size=3, output_orders=[0,1],
										 padding='SAME', name='9')
		cv9 = tf.reduce_mean(sum_magnitudes(cv9), reduction_indices=[1,2])
		return tf.nn.bias_add(cv9, biases['b9'])

def deep_bsd(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = int(opt['n_filters'])
	nf2 = int((opt['filter_gain'])*nf)
	nf3 = int((opt['filter_gain']**2)*nf)
	nf4 = int((opt['filter_gain']**3)*nf)
	bs = opt['batch_size']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'w1_1' : get_weights_dict([[3,],[2,]], opt['n_channels'], nf, std_mult=sm, name='W1_1', device=device),
			'w1_2' : get_weights_dict([[3,],[2,]], nf, nf, std_mult=sm, name='W1_2', device=device),
			'w2_1' : get_weights_dict([[3,],[2,],], nf, nf2, std_mult=sm, name='W2_1', device=device),
			'w2_2' : get_weights_dict([[3,],[2,],], nf2, nf2, std_mult=sm, name='W2_2', device=device),
			'w3_1' : get_weights_dict([[3,],[2,],], nf2, nf3, std_mult=sm, name='W3_1', device=device),
			'w3_2' : get_weights_dict([[3,],[2,],], nf3, nf3, std_mult=sm, name='W3_2', device=device),
			'w4_1' : get_weights_dict([[3,],[2,],], nf3, nf4, std_mult=sm, name='W4_1', device=device),
			'w4_2' : get_weights_dict([[3,],[2,],], nf4, nf4, std_mult=sm, name='W4_2', device=device),
			'w5_1' : get_weights_dict([[3,],[2,],], nf4, nf4, std_mult=sm, name='W5_1', device=device),
			'w5_2' : get_weights_dict([[3,],[2,],], nf4, nf4, std_mult=sm, name='W5_2', device=device),
			#'w_out' : tf.get_variable('w_out', dtype=tf.float32, shape=[1,1,(order+1)*nf4,opt['n_classes']],
			#						  initializer=tf.constant_initializer(1e-2))
		}
		
		biases = {
			'b1_2' : tf.get_variable('b1_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			'b2_2' : tf.get_variable('b2_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			'b3_2' : tf.get_variable('b3_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			'b4_2' : tf.get_variable('b4_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			'b5_2' : tf.get_variable('b5_1', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2)),
			#'b6' : tf.get_variable('b6', dtype=tf.float32, shape=[opt['n_classes']],
			#					   initializer=tf.constant_initializer(1e-2)),
			'cb1_1' : get_bias_dict(nf, order, name='cb1_1', device=device),
			'cb2_1' : get_bias_dict(nf2, order, name='cb2_1', device=device),
			'cb3_1' : get_bias_dict(nf3, order, name='cb3_1', device=device),
			'cb4_1' : get_bias_dict(nf4, order, name='cb4_1', device=device),
			'cb5_1' : get_bias_dict(nf4, order, name='cb5_1', device=device),
			'fuse' : tf.get_variable('fuse', dtype=tf.float32, shape=[1],
				initializer=tf.constant_initializer(1e-2))
		}
		
		side_weights = {
			'sw1' : tf.get_variable('sw1', dtype=tf.float32, shape=[1,1,(order+1)*nf,1],
				initializer=tf.constant_initializer(1e-2)),
			'sw2' : tf.get_variable('sw2', dtype=tf.float32, shape=[1,1,(order+1)*nf2,1],
				initializer=tf.constant_initializer(1e-2)),
			'sw3' : tf.get_variable('sw3', dtype=tf.float32, shape=[1,1,(order+1)*nf3,1],
				initializer=tf.constant_initializer(1e-2)),
			'sw4' : tf.get_variable('sw4', dtype=tf.float32, shape=[1,1,(order+1)*nf4,1],
				initializer=tf.constant_initializer(1e-2)),
			'sw5' : tf.get_variable('sw5', dtype=tf.float32, shape=[1,1,(order+1)*nf4,1],
				initializer=tf.constant_initializer(1e-2)),
			'h1' : tf.get_variable('h1', dtype=tf.float32, shape=[1,1,5,1],
				initializer=tf.constant_initializer(1e-2))
		}
		
		psis = {
			'psi1_1' : get_phase_dict(1, nf, order, name='psi1_1', device=device),
			'psi1_2' : get_phase_dict(nf, nf, order, name='psi1_2', device=device),
			'psi2_1' : get_phase_dict(nf, nf2, order, name='psi2_1', device=device),
			'psi2_2' : get_phase_dict(nf2, nf2, order, name='psi2_2', device=device),
			'psi3_1' : get_phase_dict(nf2, nf3, order, name='psi3_1', device=device),
			'psi3_2' : get_phase_dict(nf3, nf3, order, name='psi3_2', device=device),
			'psi4_1' : get_phase_dict(nf3, nf4, order, name='psi4_1', device=device),
			'psi4_2' : get_phase_dict(nf4, nf4, order, name='psi4_2', device=device),
			'psi5_1' : get_phase_dict(nf4, nf4, order, name='psi5_1', device=device),
			'psi5_2' : get_phase_dict(nf4, nf4, order, name='psi5_2', device=device),
		}
		
		nonlin = tf.nn.relu
		fm = {}
		cv = {}
		
	# Convolutional Layers
	with tf.name_scope('stage1') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1_1'], psis['psi1_1'],
				 filter_size=3, padding='VALID', name='1_1')
		cv1 = complex_nonlinearity(cv1, biases['cb1_1'], nonlin)
	
		cv2 = complex_input_rotated_conv(cv1, weights['w1_2'], psis['psi1_2'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='1_2')
		cv2 = complex_batch_norm(cv2, nonlin, phase_train, name='bn1', device=device)
		fm[1] = conv2d(stack_magnitudes(cv2), side_weights['sw1']) #, b=biases['b1_2'])
		cv[1] = cv2
	
	with tf.name_scope('stage2') as scope:
		cv3 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv3 = complex_input_rotated_conv(cv3, weights['w2_1'], psis['psi2_1'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='2_1')
		cv3 = complex_nonlinearity(cv3, biases['cb2_1'], nonlin)
	
		cv4 = complex_input_rotated_conv(cv3, weights['w2_2'], psis['psi2_2'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='2_2')
		cv4 = complex_batch_norm(cv4, nonlin, phase_train, name='bn2', device=device)
		fm[2] = conv2d(stack_magnitudes(cv4), side_weights['sw2']) #, b=biases['b2_2'])
		cv[2] = cv4
		
	with tf.name_scope('stage3') as scope:
		cv5 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv5 = complex_input_rotated_conv(cv5, weights['w3_1'], psis['psi3_1'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='3_1')
		cv5 = complex_nonlinearity(cv5, biases['cb3_1'], nonlin)
	
		cv6 = complex_input_rotated_conv(cv5, weights['w3_2'], psis['psi3_2'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='3_2')
		cv6 = complex_batch_norm(cv6, nonlin, phase_train, name='bn3', device=device)
		fm[3] = conv2d(stack_magnitudes(cv6), side_weights['sw3']) #, b=biases['b3_2'])
		cv[3] = cv6
		
	with tf.name_scope('stage4') as scope:
		cv7 = mean_pooling(cv6, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv7 = complex_input_rotated_conv(cv7, weights['w4_1'], psis['psi4_1'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='4_1')
		cv7 = complex_nonlinearity(cv7, biases['cb4_1'], nonlin)
	
		cv8 = complex_input_rotated_conv(cv7, weights['w4_2'], psis['psi4_2'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='4_2')
		cv8 = complex_batch_norm(cv8, nonlin, phase_train, name='bn4', device=device)
		fm[4] = conv2d(stack_magnitudes(cv8), side_weights['sw4']) #, b=biases['b4_2'])
		cv[4] = cv8
		
	with tf.name_scope('stage5') as scope:
		cv9 = mean_pooling(cv8, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv9 = complex_input_rotated_conv(cv9, weights['w5_1'], psis['psi5_1'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='5_1')
		cv9 = complex_nonlinearity(cv9, biases['cb5_1'], nonlin)
	
		cv10 = complex_input_rotated_conv(cv9, weights['w5_2'], psis['psi5_2'],
				 filter_size=3, output_orders=[0,1], padding='VALID', name='5_2')
		cv10 = complex_batch_norm(cv10, nonlin, phase_train, name='bn5', device=device)
		fm[5] = conv2d(stack_magnitudes(cv10), side_weights['sw5']) #, b=biases['b5_2'])
		cv[5] = cv10
		
		out = 0
	
	fms = {}
	side_preds = []
	xsh = tf.shape(x)
	with tf.name_scope('fusion') as scope:
		for key in fm.keys():
			if opt['machine'] == 'grumpy':
				fms[key] = tf.image.resize_images(fm[key], tf.pack([xsh[1], xsh[2]]))
			else:
				fms[key] = tf.image.resize_images(fm[key], xsh[1], xsh[2])
			side_preds.append(fms[key])
		side_preds = tf.concat(3, side_preds)

		fms['fuse'] = conv2d(side_preds, side_weights['h1'], b=biases['fuse'], padding='SAME')
		return fms, out, weights, psis, cv

def deep_unet(opt, x, phase_train, device='/cpu:0'):
	"""High frequency convolutions are unstable, so get rid of them"""
	# Sure layers weight & bias
	order = 1
	nf = int(opt['n_filters'])
	nf2 = int((opt['filter_gain'])*nf)
	nf3 = int((opt['filter_gain']**2)*nf)
	nf4 = int((opt['filter_gain']**3)*nf)
	bs = opt['batch_size']
	size = opt['dim']
	size2 = opt['dim2']
	
	sm = opt['std_mult']
	with tf.device(device):
		weights = {
			'd1_1' : get_weights_dict([[3,],[2,]], opt['n_channels'], nf, std_mult=sm, name='d1_1', device=device),
			'd1_2' : get_weights_dict([[3,],[2,]], nf, nf, std_mult=sm, name='d1_2', device=device),
			'd2_1' : get_weights_dict([[3,],[2,],], nf, nf2, std_mult=sm, name='d2_1', device=device),
			'd2_2' : get_weights_dict([[3,],[2,],], nf2, nf2, std_mult=sm, name='d2_2', device=device),
			'd3_1' : get_weights_dict([[3,],[2,],], nf2, nf3, std_mult=sm, name='d3_1', device=device),
			'd3_2' : get_weights_dict([[3,],[2,],], nf3, nf3, std_mult=sm, name='d3_2', device=device),
			'd4_1' : get_weights_dict([[3,],[2,],], nf3, nf4, std_mult=sm, name='d4_1', device=device),
			'd4_2' : get_weights_dict([[3,],[2,],], nf4, nf4, std_mult=sm, name='d4_2', device=device)
		}
		
		psis = {
			'd1_1' : get_phase_dict(1, nf, order, name='d1_1', device=device),
			'd1_2' : get_phase_dict(nf, nf, order, name='d1_2', device=device),
			'd2_1' : get_phase_dict(nf, nf2, order, name='d2_1', device=device),
			'd2_2' : get_phase_dict(nf2, nf2, order, name='d2_2', device=device),
			'd3_1' : get_phase_dict(nf2, nf3, order, name='d3_1', device=device),
			'd3_2' : get_phase_dict(nf3, nf3, order, name='d3_2', device=device),
			'd4_1' : get_phase_dict(nf3, nf4, order, name='d4_1', device=device),
			'd4_2' : get_phase_dict(nf4, nf4, order, name='d4_2', device=device),
			'u1_1' : get_phase_dict(nf4, nf4, order, name='u1_1', device=device),
			'u1_2' : get_phase_dict(nf4, nf4, order, name='u1_2', device=device),
			'u2_1' : get_phase_dict(nf, nf2, order, name='u2_1', device=device),
			'u2_2' : get_phase_dict(nf2, nf2, order, name='u2_2', device=device),
			'u3_1' : get_phase_dict(nf3, nf2, order, name='u3_1', device=device),
			'u3_2' : get_phase_dict(nf2, nf, order, name='u3_2', device=device),
			'u4_1' : get_phase_dict(nf2, nf, order, name='u4_1', device=device),
			'u4_2' : get_phase_dict(nf, nf, order, name='u4_2', device=device)
		}
		
		biases = {
			'b1' : get_bias_dict(nf, order, name='b1', device=device)
		}
		
		x = tf.reshape(x, tf.pack([opt['batch_size'],size,size2,3]))
		fm = {}
		
	# Convolutional Layers
	out, d1 =  down_block(True, x, weights['d1_1'], weights['d1_2'],
						  psis['psi1_1'], psis['psi1_2'], biases['b1'],
						  phase_train, name='down1', device=device)
	out, d2 =  down_block(False, out, weights['d2_1'], weights['d2_2'],
						  psis['psi2_1'], psis['psi2_2'], biases['b2'],
						  phase_train, name='down2', device=device)
	out, d3 =  down_block(False, out, weights['d3_1'], weights['d3_2'],
						  psis['psi3_1'], psis['psi3_2'], biases['b3'],
						  phase_train, name='down3', device=device)
	out, d4 =  down_block(False, out, weights['d4_1'], weights['d4_2'],
						  psis['psi4_1'], psis['psi4_2'], biases['b4'],
						  phase_train, name='down4', device=device)
	
	out =  up_block(out, d4, weights['u4_1'], weights['u4_2'], psis['u4_1'],
					psis['u4_2'], biases['u4'], phase_train, name='up4', device=device)
	out =  up_block(out, d3, weights['u3_1'], weights['u3_2'], psis['u3_1'],
					psis['u3_2'], biases['u3'], phase_train, name='up3', device=device)
	out =  up_block(out, d2, weights['u2_1'], weights['u2_2'], psis['u2_1'],
					psis['u2_2'], biases['u2'], phase_train, name='up2', device=device)
	out =  up_block(out, d1, weights['u1_1'], weights['u1_2'], psis['u1_1'],
					psis['u1_2'], biases['u1'], phase_train, name='up1', device=device)
