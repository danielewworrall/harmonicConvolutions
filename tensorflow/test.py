'''Test the gConv script'''

import os
import sys
import time

import numpy as np
import tensorflow as tf

import input_data

from gConv2 import *

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 1e-3
training_iters = 200000
batch_size = 40
display_step = 10
save_step = 100

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def lieConv2d(X, n_filters, b, name):
	# Lie Conv 2D wrapper, with bias and relu activation
	phi, y = gConv(X, 3, n_filters, name=name)
	#x = tf.concat(3, [phi, y])
	x = tf.nn.bias_add(y, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	# Convolution Layer
	conv1 = lieConv2d(x, 32, biases['bc1'], name='gc1')
	conv1_ = maxpool2d(conv1, k=2)
	
	# Convolution Layer
	conv2 = lieConv2d(conv1_, 32, biases['bc2'], name='gc2')
	conv2 = maxpool2d(conv2, k=2)
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	
	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

# Store layers weight & bias
weights = {
    # fully connected, 6*6*32 inputs, 1024 outputs
    'wd1': tf.Variable(tf.sqrt(6.0/(2176.))*tf.random_normal([6*6*32, 1024]), name='W'),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.sqrt(6.0/(1034.))*tf.random_normal([1024, n_classes]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([32])),
	'bd1': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
gvs = opt.compute_gradients(cost)
clip = 2.
capped_gvs = [(tf.clip_by_value(gv[0], -clip, clip),gv[1]) for gv in gvs]
#capped_gvs = [(gv[0],gv[1]) for gv in gvs]
optimizer = opt.apply_gradients(capped_gvs)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope('model'):
	tf.scalar_summary('accuracy', accuracy)

# Create orthogonalization routine
Q_var = []
W_var = []
orthogonalize_ops = []
for var in tf.all_variables():
	if 'Adam' not in var.name:
		if '_Q' in var.name:
			Q_var.append(var)
			print var.name
		if 'W' in var.name:
			W_var.append(var)
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
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		
		Q1, Q2 = sess.run(Q_var)
		Q1 = np.reshape(ortho(np.reshape(Q1, [9,9])), [3,3,1,9])
		Q2 = np.reshape(ortho(np.reshape(Q2, [9,9])), [3,3,1,9])
		sess.run(orthogonalize_ops, feed_dict={Q_1 : Q1, Q_2 : Q2})
		
		feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout}
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
