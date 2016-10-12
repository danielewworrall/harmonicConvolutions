'''Test the gConv script'''

import os
import sys
import time

import numpy as np
import scipy.linalg as scilin
import tensorflow as tf

import input_data
import nesterov

from gConv2 import *
from matplotlib import pyplot as plt

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def lieConv2d(X, n_filters, b_y, b_phi, name):
	# Lie Conv 2D wrapper, with bias and relu activation
	phi, y = gConv(X, 3, n_filters, name=name)
	
	y = tf.nn.relu(tf.nn.bias_add(y,b_y))
	phi = modulus(tf.nn.bias_add(phi,b_phi), 2*np.pi)
	
	return tf.concat(3, [phi, y])

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, drop_prob, n_filters):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	# Convolution Layer
	conv1 = lieConv2d(x, n_filters, biases['by1'], biases['bphi1'], name='gc1')
	conv1_ = maxpool2d(conv1, k=2)
	
	# Convolution Layer
	conv2 = lieConv2d(conv1_, n_filters, biases['by2'], biases['bphi2'], name='gc2')
	conv2 = maxpool2d(conv2, k=2)
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	#conv2 = tf.nn.dropout(conv2, drop_prob)
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, drop_prob)
	
	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

def geodesic_flow(W, dJ_dW, lr):
	"""
	Take gradient descent step along Lie manifold using Plumbley (2005)
	
	W : Matrix in SO(N)
	dJ_dW : unconstained gradient update for W
	lr : learning rate
	"""
	X = np.dot(dJ_dW,W.T)
	dJ_dB = X - X.T
	update_rotation = scilin.expm(-lr*dJ_dB)
	return np.dot(update_rotation,W)

def run():
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	
	# Parameters
	lr = 1e-2
	training_iters = 200000
	batch_size = 40
	display_step = 10
	save_step = 100
	
	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)
	dropout = 0.75 # Dropout, probability to keep units
	n_filters = 20
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	
	# Store layers weight & bias
	weights = {
		
		# fully connected, 6*6*32 inputs, 1024 outputs
		'wd1': tf.Variable(tf.sqrt(6.0/(1652.))*tf.random_normal([6*6*n_filters*2, 500]), name='W'),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.sqrt(6.0/(510.))*tf.random_normal([500, n_classes]))
	}
	
	biases = {
		'by1': tf.Variable(tf.random_normal([n_filters])),
		'by2': tf.Variable(tf.random_normal([n_filters])),
		'bphi1': tf.Variable(tf.random_normal([n_filters])),
		'bphi2': tf.Variable(tf.random_normal([n_filters])),
		'bd1': tf.Variable(tf.random_normal([500])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}
	
	# Construct model
	pred = conv_net(x, weights, biases, keep_prob, n_filters)
	
	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.)
	gvs = opt.compute_gradients(cost)
	clip = 2.
	new_gvs = [(tf.clip_by_value(g, -clip, clip), v) for (g,v) in gvs]
	optimizer = opt.apply_gradients(new_gvs)
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	with tf.name_scope('model'):
		tf.scalar_summary('accuracy', accuracy)
	
	# Create orthogonalization routine
	Q_var = []
	orthogonalize_ops = []
	for var in tf.all_variables():
		if 'Momentum' not in var.name:
			if '_Q' in var.name:
				Q_var.append(var)
				print var.name
	Q_1 = tf.placeholder(tf.float32, [3,3,1,9], 'Q_1')
	Q_2 = tf.placeholder(tf.float32, [3,3,1,9], 'Q_2')
	orthogonalize_ops.append(Q_var[0].assign(Q_1))
	orthogonalize_ops.append(Q_var[1].assign(Q_2))
	
	def ortho(Q):
		U, __, V = np.linalg.svd(Q)
		return np.dot(U,V)
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Create a saver
	saver = tf.train.Saver()
	merged = tf.merge_all_summaries()
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		writer = tf.train.SummaryWriter('./logs/', sess.graph_def)
		step = 1
		start = time.time()
		# Keep training until reach max iterations
		while step * batch_size < training_iters:
			lr_ = lr / np.sqrt(step + 1.)
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			
			# Orthogonalize Q
			Q1, Q2 = sess.run(Q_var)
			Q1 = np.reshape(ortho(np.reshape(Q1, [9,9])), [3,3,1,9])
			Q2 = np.reshape(ortho(np.reshape(Q2, [9,9])), [3,3,1,9])
			sess.run(orthogonalize_ops, feed_dict={Q_1 : Q1, Q_2 : Q2})
			
			# Optimize
			feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout,
						 learning_rate : lr_}
			sess.run(optimizer, feed_dict=feed_dict)
			
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				feed_dict = {x: batch_x, y: batch_y, keep_prob: 1.}
				summary, loss, acc = sess.run([merged, cost, accuracy],
					feed_dict=feed_dict)
				print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc)
				writer.add_summary(summary, step)
				print "Time: " + str((time.time()-start)/np.maximum(step,1.))
			step += 1
			
			if step % save_step == 0:
				saver.save(sess, './checkpoints/model.ckpt', global_step=step)
		print "Testing"
		
		# Test accuracy
		tacc = 0.
		for i in xrange(200):
			feed_dict={ x: mnist.test.images[50*i:50*(i+1)],
					   y: mnist.test.labels[50*i:50*(i+1)], keep_prob: 1.}
			tacc += sess.run(accuracy, feed_dict=feed_dict)
		print('Test accuracy: %f' % (tacc/200.,))

def restore_weights():
	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = 1.
	
	# Store layers weight & bias
	weights = {
		# fully connected, 6*6*32 inputs, 1024 outputs
		'wd1': tf.Variable(tf.sqrt(6.0/(1652.))*tf.random_normal([6*6*n_filters*2, 500]), name='W'),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.sqrt(6.0/(510.))*tf.random_normal([500, n_classes]))
	}
	
	biases = {
		'by1': tf.Variable(tf.random_normal([16])),
		'by2': tf.Variable(tf.random_normal([16])),
		'bphi1': tf.Variable(tf.random_normal([16])),
		'bphi2': tf.Variable(tf.random_normal([16])),
		'bd1': tf.Variable(tf.random_normal([500])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}
	
	# Construct model
	pred = conv_net(x, weights, biases, keep_prob)
	
	# Create a saver
	saver = tf.train.Saver()
	
	# Launch the graph
	with tf.Session() as sess:
		saver.restore(sess, './checkpoints/model.ckpt-1300')
		print('Weights restored')
		
		V_op = []
		for var in tf.all_variables():
			if 'Adam' not in var.name:
				if '_V' in var.name:
					V_op.append(var)
		V_eval = sess.run(V_op)
		
		for V in V_eval:
			V = np.reshape(V,[3,3,32])
			for i in xrange(32):
				fig = plt.figure(1)
				plt.imshow(V[:,:,i], cmap='gray', interpolation='nearest')
				plt.show()

if __name__ == '__main__':
	run()
	#restore_weights()
