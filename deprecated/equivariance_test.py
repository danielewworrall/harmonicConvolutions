"""Equivariance tests"""

import os
import sys
import time
sys.path.append('../')

import numpy as np
import tensorflow as tf

import harmonic_network_lite as hl


def test_forward_invariance_90():
   """Test invariance to 90 rotations of the input"""
   tf.reset_default_graph()
   x = tf.placeholder(tf.float32, [4,3,3,1,1,1])
   y = hl.conv2d(x, 1, 3, name='conv_forward_invariance_90')
   inv = hl.stack_magnitudes(y)

   X1 = np.random.randn(1,3,3,1,1,1)
   X2 = np.transpose(X1, (0,2,1,3,4,5))[:,:,::-1,:,:,:]
   X3 = np.transpose(X2, (0,2,1,3,4,5))[:,:,::-1,:,:,:]
   X4 = np.transpose(X3, (0,2,1,3,4,5))[:,:,::-1,:,:,:]
   X = np.concatenate((X1,X2,X3,X4), axis=0)

   with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      Y, Inv = sess.run([y, inv], feed_dict={x: X})

   Y = np.squeeze(Y)
   Inv = np.squeeze(Inv)
   # Look at the difference in order 0 vectors...should be EXACTLY 0
   for i in xrange(4):
      for j in xrange(i):
         assert np.amax(np.abs(Y[j,0,:] - Y[i,0,:])) < 1e-5
         print np.amax(np.abs(Y[j,0,:] - Y[i,0,:]))
   print
   # Look at the difference in magnitudes
   for i in xrange(4):
      for j in xrange(i):
         assert np.amax(np.abs(Inv[j,:] - Inv[i,:])) < 1e-5
