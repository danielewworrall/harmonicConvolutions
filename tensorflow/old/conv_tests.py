"""Convolution tests"""

import os
import sys
import time

import numpy as np
import tensorflow as tf

from gConv2 import *
from matplotlib import pyplot as plt


def rotation_stack_test():
	"""This test should return the filter at 4 orientations"""
	v = tf.placeholder(tf.float32, [3,3,1,2], name='V')
	v_ = get_rotation_stack(v)
	
	V = np.random.randn(3,3,1,2)
	
	with tf.Session() as sess:
		V_ = sess.run(v_, feed_dict={v : V})
		
	V_ = np.squeeze(V_)
	V_ = np.reshape(V_, [3,-1], order='F')
	
	fig = plt.figure(1)
	plt.imshow(V_, cmap='gray', interpolation='nearest')
	plt.show()



if __name__ == '__main__':
	rotation_stack_test()