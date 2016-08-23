'''Test the gConv script'''

import os
import sys
import time

import numpy as np
import tensorflow as tf

from gConv2 import *

x = tf.placeholder("float", [5,6,6,3], name='x')
z = gConv(x, 3, 7, name='gConv')

X = np.random.randn(5,6,6,3).astype(np.float32)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	Z = sess.run(z, feed_dict={x : X})
	print Z[0].shape, Z[1].shape