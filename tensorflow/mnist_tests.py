'''MNIST tests'''

'''Test the gConv script'''

import os
import sys
import time

import numpy as np
import scipy.linalg as scilin
import tensorflow as tf

import input_data

from gConv2 import *
from matplotlib import pyplot as plt


# Create model
def conv_net(x, weights, biases, drop_prob, n_filters):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	# Convolution Layer
	gc1 = gConv(x, weights['w1'], biases['b1'], name='gc1')
	gc1_ = tf.nn.relu(maxpool2d(gc1, k=2))
	
	# Convolution Layer
	gc2 = gConv(gc1_, weights['w2'], biases['b2'], name='gc2')
	gc2_ = tf.nn.relu(maxpool2d(gc2, k=2))

	# Fully connected layer
	fc3 = tf.reshape(gc2_, [-1, weights['w3'].get_shape().as_list()[0]])
	fc3 = tf.nn.bias_add(tf.matmul(fc3, weights['w3']), biases['b3'])
	fc3 = tf.nn.relu(fc3)
	# Apply Dropout
	fc3 = tf.nn.dropout(fc3, drop_prob)
	
	# Output, class prediction
	out = tf.nn.bias_add(tf.matmul(fc3, weights['out']), biases['out'])
	return out

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def get_weights(filter_shape, W_init=None, name='W'):
	if W_init == None:
		stddev = np.sqrt(2.0 / np.prod(filter_shape[:2]))
		W_init = tf.random_normal(filter_shape, stddev=stddev)
	return tf.Variable(W_init, name=name)

def run():
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	
	# Parameters
	lr = 1e-2
	training_iters = 200000
	batch_size = 500
	display_step = 10
	save_step = 100
	
	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)
	dropout = 0.75 # Dropout, probability to keep units
	n_filters = 16
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) 
	
	# Store layers weight & bias
	weights = {
		'w1' : get_weights([3,3,1,n_filters], name='W1'),
		'w2' : get_weights([3,3,n_filters*4,n_filters], name='W2'),
		'w3' : get_weights([n_filters*4*6*6,500], name='W3'),
		'out': get_weights([500, n_classes], name='W4')
	}
	
	biases = {
		'b1': tf.Variable(tf.constant(1e-2, shape=[n_filters])),
		'b2': tf.Variable(tf.constant(1e-2, shape=[n_filters])),
		'b3': tf.Variable(tf.constant(1e-2, shape=[500])),
		'out': tf.Variable(tf.constant(1e-2, shape=[n_classes]))
	}
	
	# Construct model
	pred = conv_net(x, weights, biases, keep_prob, n_filters)
	
	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(cost)
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while step * batch_size < training_iters:
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			
			# Optimize
			feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout}
			sess.run(optimizer, feed_dict=feed_dict)
			
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				feed_dict = {x: batch_x, y: batch_y, keep_prob: 1.}
				loss, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
				print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc)
			step += 1
			
		print "Testing"
		
		# Test accuracy
		tacc = 0.
		for i in xrange(200):
			feed_dict={ x: mnist.test.images[50*i:50*(i+1)],
					   y: mnist.test.labels[50*i:50*(i+1)], keep_prob: 1.}
			tacc += sess.run(accuracy, feed_dict=feed_dict)
		print('Test accuracy: %f' % (tacc/200.,))


if __name__ == '__main__':
	run()
