'''Analyze the gradient properties of H-convs'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import tensorflow as tf

import harmonic_network_lite as hn_lite


def main():
	nr = 5
	k = 5
	N = 100
	with tf.device('/gpu:0'):
		x = tf.placeholder(tf.float32, [N,k,k,2,1,1], name='x')
		y = hn_lite.conv2d(x, 10, k, n_rings=nr)
	
	X = np.random.randn(N,k,k,2,1,1)
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		Y = sess.run(y, feed_dict={x: X})
	
	print Y.shape


if __name__ == '__main__':
	main()