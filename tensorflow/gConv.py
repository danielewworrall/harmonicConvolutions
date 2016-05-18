'''MLP example'''

import os
import sys
import time

import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Metaparameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Architectural Hyperparameters
n_hid1 = 676
n_hid2 = 676
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _Q, _weights, _biases):
    cv1 = tf.nn.relu(tf.nn.conv2d(_X, _Q, [1,1,1,1], "VALID"))
    reshape = tf.reshape(cv1, [-1, n_hid1*9])
    fc2 = tf.nn.relu(tf.add(tf.matmul(reshape, _weights['h2']), _biases['b2']))
    return tf.matmul(fc2, _weights['out']) + _biases['out']

# Parameter storage
Q = tf.Variable(tf.random_normal([3,3,1,9], mean=0.0, stddev=0.05), name="Q") # Need to change this shape!!!!

weights = {
    'h2': tf.Variable(tf.random_normal([n_hid1*9, n_hid2])),
    'out': tf.Variable(tf.random_normal([n_hid2, n_classes]))
}
biases = {
    'b2': tf.Variable(tf.random_normal([n_hid2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, Q, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) # SGD

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
            batch_xs = batch_xs.reshape([-1, 28, 28, 1])
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
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
    print "Accuracy:", accuracy.eval({x: mnist.test.images.reshape([-1, 28, 28, 1]), y: mnist.test.labels})