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

def rot_mat_low(theta):
	"""Lowest harmonic only"""
	R = np.zeros((9,9))
	R[0,0] = np.cos(theta)
	R[1,1] = np.cos(theta)
	R[1,0] = np.sin(theta)
	R[0,1] = -np.sin(theta)
	R[8,8] = 1
	return R

def gen_data(X, N, Q):
	X_ = np.dot(Q,X)
	# Get rotation
	theta = np.linspace(0, 2*np.pi, N)
	Y = []
	Ylow = []
	for t in theta:
		R = rot_mat(t)
		Rlow = rot_mat_low(t)
		Y.append(np.dot(Q.T,np.dot(R,X_)))
		Ylow.append(np.dot(Q.T,np.dot(Rlow,X_)))
	Y = np.vstack(Y)
	Ylow = np.vstack(Ylow)
	
	return Y, Ylow

def get_Q(n):
	Q = np.random.randn(9,9)
	U, __, V = np.linalg.svd(Q)
	return np.dot(U,V)

def gConv_test():
	"""Test gConv"""
	N = 360
	X = np.random.randn(9)
	Q = get_Q(9)
	X, Xlow = gen_data(X, N, Q)
	X = np.reshape(X, [N,3,3,1])
	Q = np.transpose(Q)
	Q = np.reshape(Q, [3,3,1,9])
	
	# tf conv
	x = tf.placeholder('float', [None,3,3,1], name='x')
	q = tf.placeholder('float', [3,3,1,9], name='q')
	y = gConv(x, 3, 1, q, name='gc')
	
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		A, Y, V_ = sess.run(y, feed_dict={x : X, q : Q})
	
	V_ = np.squeeze(V_)
	Yrlow = np.dot(Xlow,V_)
	Yr = np.dot(np.reshape(np.squeeze(X),[-1,9]),V_)
	
	fig = plt.figure(1)
	plt.plot(np.squeeze(Yrlow), 'g')
	plt.plot(np.squeeze(Yr), 'r')
	plt.scatter(360-180*A[0,...]/np.pi,Y[0])
	plt.show()

def gConv_polar_test():
	"""Test gConv"""
	N = 1
	X = np.random.randn(9,N)
	X = np.reshape(X, [N,3,3,1])
	
	# tf conv
	x = tf.placeholder('float', [None,3,3,1], name='x')
	y = gConv_polar(x, 3, 2, name='gc')
	
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		A, Y = sess.run(y, feed_dict={x : X})
	
	print A.shape, Y.shape
	
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

def grad_atan2_test():
	num=200
	x = tf.placeholder('float', [num], name='x')
	y = tf.placeholder('float', [num], name='y')
	
	z = atan2(y, x)
	
	g = tf.gradients(z, y)
	
	Y = np.linspace(-10.,10.,num=num)
	X = 0.3*np.ones_like(Y)
	with tf.Session() as sess:
		Z, G = sess.run([z, g], feed_dict={x : X, y : Y})
		print sess.run([z, g], feed_dict={x : 0.*np.ones((200)), y : 0.*np.ones((200))})
	
	Y = np.squeeze(Y)
	Z = np.squeeze(np.asarray(Z))
	G = np.squeeze(np.asarray(G))
	fig = plt.figure(1)
	plt.plot(Y, Z, 'b')
	plt.plot(Y, G, 'r')
	plt.show()

def zConv_test():
	"""Manually set the angle to zero, and see if conv2d is equivalent"""
	x_shape = [10,13,12,5]
	filter_size = 3
	n_filters = 10
	f_shape = [filter_size,filter_size,x_shape[3],n_filters]
	x = tf.placeholder('float', x_shape, name='x')
	v = tf.placeholder('float', f_shape, name='f')
	q = tf.placeholder('float', [3,3,1,9], name='q')
	y = gConv(x, filter_size, n_filters, name='gc')
	z = tf.nn.conv2d(x, v, strides=(1,1,1,1), padding='VALID')
	
	X = np.random.randn(x_shape[0],x_shape[1],x_shape[2],x_shape[3])

	# Get Q variables for evaluationg
	Qvar = []
	for var in tf.all_variables():
		if '_Q' in var.name:
			Qvar.append(var)
	Qop = Qvar[0].assign(q)

	def orth(W):
		U, __, V = np.linalg.svd(W)
		return np.dot(U,V)

	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		
		# Orthogonalize Q before forward propping
		Q = sess.run(Qvar)
		Q = orth(Q[0].reshape([9,9])).reshape([3,3,1,9])
		sess.run(Qop, feed_dict={q : Q})
		Phi, Y, V = sess.run(y, feed_dict={x : X})
		
	V = V.reshape([filter_size,filter_size,x_shape[3],n_filters])
	print(V.shape)
	print(f_shape)
	with tf.Session() as sess:
		Z = sess.run(z, feed_dict={x : X, v : V})
		
	print np.sum((Y-Z)**2)
	
	# Gradients
	g_G = tf.gradients(y[1], x)
	g_Z = tf.gradients(z, x)
	mse = tf.reduce_sum(tf.pow(g_G[0]-g_Z[0],2.))
	
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		sess.run(Qop, feed_dict={q : Q})
		MSE = sess.run(mse, feed_dict={x : X, v : V})
	print(MSE) 

def gConv_grad_test():
	N = 360
	X = np.random.randn(9)
	Q = get_Q(9)
	X, Xlow = gen_data(X, N, Q)
	X = np.maximum(X,0.)
	X = np.reshape(X, [N,3,3,1])
	#Q = np.transpose(Q)
	Q_ = get_Q(9)
	Q = np.reshape(Q_, [3,3,1,9])
	
	# tf conv
	x = tf.placeholder('float', [None,3,3,1], name='x')
	q = tf.placeholder('float', [3,3,1,9], name='q')
	y = gConv(x, 3, 1, q, name='gc')
	f = tf.reduce_sum(tf.pow(tf.nn.relu(y[1]), 2))
	g = tf.gradients(f, q)
	
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		A, Y = sess.run(y, feed_dict={x : X, q : Q})
		G = sess.run(g, feed_dict={x : X, q : Q})
	
	print np.sum(np.isnan(G[0]))
	print G[0]
	
	fig = plt.figure(1)
	plt.plot(np.squeeze(G[0][2,2,0,:]))
	plt.show()

def grad_descent_test():
	"""Discriminate between two shapes"""
	A = np.eye(3)
	B = 0.*(A + np.fliplr(A) > 0.)
	
	N = 360
	Q = get_Q(9)
	A, __ = gen_data(A.reshape([9,]), N, Q)
	B, __ = gen_data(B.reshape([9,]), N, Q)
	
	# Parameters
	lr = 1e-3
	training_iters = 200000
	batch_size = 1000
	display_step = 10
	
	# Network Parameters
	n_input = 9 # MNIST data input (img shape: 28*28)
	n_classes = 2 # MNIST total classes (0-9 digits)
	n_filters = 2
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])
	learning_rate = tf.placeholder(tf.float32)
	
	biases = {
		'by': tf.Variable(tf.random_normal([n_filters])),
		'bphi': tf.Variable(tf.random_normal([n_filters]))
	}
	
	# Construct model
	x_ = tf.reshape(x, shape=[-1, 3, 3, 1])
	phi, resp = gConv(x_, 3, n_filters, name='gc')
	pred = tf.nn.relu(tf.nn.bias_add(resp,biases['by']))
	pred = tf.reshape(pred, [-1,2])
	#phi = modulus(tf.nn.bias_add(phi,biases['b_phi']), 2*np.pi)
	
	
	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.)
	gvs = opt.compute_gradients(cost)
	clip = 2.
	new_gvs = []
	print gvs
	for g, v in gvs:
		if g is not None:
			new_gvs.append((tf.clip_by_value(g, -clip, clip), v))
		else:
			new_gvs.append((g, v))
	optimizer = opt.apply_gradients(new_gvs)
	#optimizer = opt.apply_gradients(gvs)
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	# Create orthogonalization routine
	Q_var = []
	orthogonalize_ops = []
	for var in tf.all_variables():
		if 'Momentum' not in var.name:
			if '_Q' in var.name:
				Q_var.append(var)
				print var.name
	Q_1 = tf.placeholder(tf.float32, [3,3,1,9], 'Q_1')
	orthogonalize_ops.append(Q_var[0].assign(Q_1))
	
	def ortho(Q):
		U, __, V = np.linalg.svd(Q)
		return np.dot(U,V)
	
	def sample_data(A, B, N):
		bit = (np.random.rand(N) > 0.5).reshape([-1,1])
		angle = np.random.randint(360, size=N)
		A_ = A[angle,:]
		B_ = B[angle,:]
		Z = bit*A_ + (1.-bit)*B_
		bit = np.hstack([bit,1-bit])
		return (Z, bit)
			
	# Initializing the variables
	init = tf.initialize_all_variables()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while step < training_iters:
			lr_ = lr / np.sqrt(step + 1.)
			batch_x, batch_y = sample_data(A, B, batch_size)
			
			# Orthogonalize Q
			Q1 = sess.run(Q_var)
			Q1 = np.reshape(ortho(np.reshape(Q1, [9,9])), [3,3,1,9])
			sess.run(orthogonalize_ops, feed_dict={Q_1 : Q1})
			
			# Optimize
			feed_dict = {x: batch_x, y: batch_y, learning_rate : lr_}
			__, acc = sess.run([optimizer, accuracy], feed_dict=feed_dict)
			if step % display_step == 0:
				print str(step) + '   ' + str(acc)
			step += 1

def complex_basis_test():
	k = 20
	X, Y = get_complex_basis(k=k, n=2, wrap=-1.)
	R = tf.squeeze(tf.sqrt(X**2 + Y**2))
	X = tf.squeeze(X)/R
	Y = tf.squeeze(Y)/R
	
	plt.figure(1)
	plt.ion()
	plt.show()
	N = 18
	for i in xrange(N):
		plt.cla()
		x = tf.cos(i*np.pi/9.)*X[:,:,0] + tf.sin(i*np.pi/9.)*X[:,:,1]
		y = tf.cos(i*np.pi/9.)*Y[:,:,0] + tf.sin(i*np.pi/9.)*Y[:,:,1]
		
		with tf.Session() as sess:
			R_, X_, Y_ = sess.run([R, x, y])
		
		plt.imshow(R_[:,:,0])
		plt.quiver(X_, Y_)
		plt.draw()
		raw_input(i*np.pi/9)

if __name__ == '__main__':
	#get_rotation_as_vectors_test()
	#mutual_tile_test()
	#channelwise_conv2d_test()
	#gConv_test()
	#gConv_polar_test()
	#grad_atan2_test()
	#gConv_grad_test()
	#grad_descent_test()
	#zConv_test()
	complex_basis_test()































