'''Grad sqrt()'''

import os
import sys
import time

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt


def mod(x):
	return tf.sqrt(tf.maximum(tf.square(x),1e-8))

def inv_mod(x):
	return tf.rsqrt(tf.maximum(tf.square(x),1e-8))

def relu(x, b):
	return tf.nn.relu(mod(x) - b) * x * inv_mod(x)

x = tf.placeholder(tf.float32, [None,], name='x')
y = x
for i in xrange(100):
	y = relu(y, i*0.0001)
g = tf.gradients(y, x)

X = np.linspace(-1., 1., 1001)

with tf.Session() as sess:
	G = sess.run(g, feed_dict={x: X})

print np.sum(G[0])
plt.figure()
plt.plot(X, G[0])
plt.show()