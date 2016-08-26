'''Units tests'''

import os
import sys
import time

import numpy as np
import scipy.ndimage.interpolation as scint
import scipy.linalg as scilin
import scipy.signal as scisig
import tensorflow as tf

import input_data

from gConv2 import *
from matplotlib import pyplot as plt

def rot_mat(theta):
	R = np.zeros((9,9))
	for i in xrange(4):
		R[2*i,2*i] = np.cos((i+1)*theta)
		R[2*i+1,2*i+1] = np.cos((i+1)*theta)
		R[2*i+1,2*i] = np.sin((i+1)*theta)
		R[2*i,2*i+1] = -np.sin((i+1)*theta)
	R[8,8] = 1
	return R

def gen_data(X, N, Q):
	X_ = np.dot(Q,X)
	# Get rotation
	theta = np.linspace(0, 2*np.pi, N)
	Y = []
	for t in theta:
		R = rot_mat(t)
		Y.append(np.dot(Q.T,np.dot(R,X_)))
	return np.vstack(Y)

def get_Q(n):
	Q = np.random.randn(9,9)
	U, __, V = np.linalg.svd(Q)
	return np.dot(U,V)

def gConv_test():
	"""Test gConv"""
	N = 360
	X = np.random.randn(9)
	Q = get_Q(9)
	#Q = np.eye(9)
	X = gen_data(X, N, Q)
	X = np.reshape(X, [N,3,3,1])
	Q = np.transpose(Q)
	Q = np.reshape(Q, [3,3,1,9])
	#Q = np.transpose(Q, [1,0,2,3])
	
	# tf conv
	x = tf.placeholder('float', [None,3,3,1], name='x')
	q = tf.placeholder('float', [3,3,1,9], name='q')
	y = gConv(x, 3, 1, q, name='gc')
	
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		A, Y = sess.run(y, feed_dict={x : X, q : Q})
	
	fig = plt.figure(1)
	plt.plot(np.squeeze(Y))
	plt.ylim([-0.05, 0.05])
	plt.show()
	
	
def channelwise_conv2d_test():
	"""Test channelwise conv2d"""
	Q = np.reshape(get_Q(9), (3,3,1,9))
	h = 3
	w = 3
	b = 20
	c = 10
	X = np.random.randn(b,h,w,c)
	
	# tf conv
	x = tf.placeholder('float', [None,h,w,c], name='x')
	q = tf.placeholder('float', [3,3,1,9], name='q')
	y = channelwise_conv2d(x, q)
	
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		Y = sess.run(y, feed_dict={x : X, q : Q})
	
	# scipy conv
	X_ = np.transpose(X, [3,0,1,2])
	Y_ = np.zeros((9,c,b,h-2,w-2))
	for i in xrange(X_.shape[0]):
		for j in xrange(X_.shape[1]):
			for k in xrange(Q.shape[-1]):
				Q_ = np.squeeze(Q[...,k])
				Y_[k,i,j,...] = scisig.correlate2d(X_[i,j,...],Q_,mode='valid')
	print Y_.shape
	
	# Dense mult
	Q_ = np.transpose(Q, [3,0,1,2])
	Q_ = np.reshape(Q_, [9,9])
	X_ = np.transpose(X, [1,2,3,0])	# [b,h,w,c] -> [h,w,c,b]
	X_ = np.reshape(X_, [9,c*b])		# [hw,cb]
	Y2 = np.dot(Q_, X_)
	Y2 = np.reshape(Y2, [9,c,b,1,1])
	
	print('Channel-wise conv2d test')
	error = np.sum((Y - Y_)**2)
	print('Error: %f' % (error,))
	error2 = np.sum((Y - Y2)**2)
	print('Error2: %f' % (error2,))

def mutual_tile_test():
	u = tf.placeholder('float', [3,1], name='u')
	v = tf.placeholder('float', [1,4], name='v')
	u_, v_ = mutual_tile(u,v)
	
	U = np.random.randn(3,1)
	V = np.random.randn(1,4)
	
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		U_, V_ = sess.run([u_,v_], feed_dict={u : U, v : V})
		
	print U
	print V
	print
	print U_
	print
	print V_

def get_rotation_as_vectors_test():
	phi = tf.placeholder('float', [4,], name='phi')
	Rcos, Rsin = get_rotation_as_vectors(phi,3)
	Phi = np.pi*np.asarray([1/4.,1/3.,1/2.,1.])
	
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		Rcos_, Rsin_ = sess.run([Rcos, Rsin], feed_dict={phi : Phi})
		
	print Rcos_
	print
	print Rsin_



if __name__ == '__main__':
	#get_rotation_as_vectors_test()
	#mutual_tile_test()
	#channelwise_conv2d_test()
	gConv_test()