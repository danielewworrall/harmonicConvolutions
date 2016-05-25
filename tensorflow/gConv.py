'''MLP example'''

import os
import sys
import time

import numpy as np
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Metaparameters
learning_rate = 0.00001
training_epochs = 150
batch_size = 50
display_step = 1

# Architectural Hyperparameters
n_in = 784
n_hid1 = 676*5
n_hid2 = 676
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases, _Q):
    gc1 = tf.nn.relu(gConv(_X, Q['Q1'], W['W1']))
    cv1 = tf.reshape(gc1, [-1,n_hid1*5])
    fc2 = tf.nn.relu(tf.add(tf.matmul(cv1, _weights['h2']), _biases['b2']))
    return tf.matmul(fc2, _weights['out']) + _biases['out']

def gConv(_X, _Q, _W, eps=1e-6):
    # Compute the projection of X and W into Q-space
    Qx = tf.nn.depthwise_conv2d(_X, _Q, strides=(1,1,1,1), padding="VALID")
    Qw = tf.matmul(tf.transpose(tf.reshape(_Q, [9,9])), _W)    # Each col. a filter
    # Find the subvector angles for the rotations
    # Segment_xxx performs op xxx on segmentation of first dimension
    Qx = tf.transpose(Qx, perm=[3,0,1,2])
    wX = tf.reshape(Qw, [9,1,1,1])* Qx
    normQx = tf.sqrt(tf.segment_sum(tf.pow(Qx,2), [0,0,1,1,2,2,3,3,4]))
    normQw = tf.sqrt(tf.segment_sum(tf.pow(Qw,2), [0,0,1,1,2,2,3,3,4]))
    dot = tf.segment_sum(wX, [0,0,1,1,2,2,3,3,4])
    print normQx
    print normQw
    normDot = tf.truediv(tf.truediv(dot, normQx + eps), tf.reshape(normQw, [5,1,1,1]) + eps)
    # normDot is a tensor of dotProducts, we can return the angle using acos
    return tf.transpose(normDot, perm=[1,2,3,0])

weights = {
    'h2': tf.Variable(tf.random_normal([n_hid1, n_hid2], mean=0.06)),
    'out': tf.Variable(tf.random_normal([n_hid2, n_classes], mean=0.03))
}
biases = {
    'b2': tf.Variable(tf.random_normal([n_hid2], mean=0.1, stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], mean=0.1, stddev=0.01))
}
Q = {
    'Q1' : tf.Variable(tf.random_normal([3,3,1,9], mean=0., stddev=0.06)),
    'Q2' : tf.Variable(tf.random_normal([3,3,1,9], mean=0., stddev=0.06))
}
W = {
    'W1' : tf.Variable(tf.random_normal([9,1], mean=1., stddev=0.06)),
    'W2' : tf.Variable(tf.random_normal([9,1], mean=1., stddev=0.06))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, Q)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss) # SGD

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        start = time.time()
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            batch_xs = batch_xs.reshape([-1,28,28,1])
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(loss , feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "[%04d]" % (epoch+1)
            print "\tTime: %f" % (time.time()-start)
            print "\tCost:", "{:.9f}".format(avg_cost)
            
        # Calculate accuracy
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print "\tAccu:", accuracy.eval({x: mnist.test.images.reshape([-1, 28, 28, 1]), y: mnist.test.labels})
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})