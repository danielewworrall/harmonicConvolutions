'''Old tests'''

import os
import sys
import time

def forward():
	"""Experiment to demonstrate the equivariance of the convolution"""
	#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')

	# Parameters
	lr = 1e-2
	batch_size = 500
	dataset_size = 50000
	valid_size = 5000
	n_epochs = 150
	display_epoch = 10
	save_step = 100
	test_rot = True
	
	# Network Parameters
	n_input = 784 		# MNIST data input (img shape: 28*28)
	n_classes = 10		# MNIST total classes (0-9 digits)
	dropout = 0.75 		# Dropout, probability to keep units
	n_filters = 10
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, n_input])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	
	# Construct model
	nlx, nly = single_steer(x, n_filters, n_classes, batch_size, phase_train)
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		epoch = 0
		start = time.time()
		# Keep training until reach max iterations
		generator = minibatcher(mnist_train['x'], mnist_train['y'], batch_size, shuffle=True)
		cost_total = 0.
		acc_total = 0.
		vacc_total = 0.
		
		for i, batch in enumerate(generator):
			batch_x, batch_y = batch
			n=500
			batch_x = ring_rotation(batch_x[4,:], n=n)
			
			feed_dict = {x: batch_x, phase_train : False}
			nlx_, nly_ = sess.run([nlx, nly], feed_dict=feed_dict)
			
			R = np.sqrt(nlx_**2 + nly_**2 + 1e-6)
			nlx_ = nlx_/R
			nly_ = nly_/R
			#R = np.reshape(R, [-1,10,1])
			#nlx_ = np.reshape(nlx_, [-1,10,1])
			#nly_ = np.reshape(nly_, [-1,10,1])
			
			plt.figure(1)
			plt.ion()
			plt.show()
			for j in xrange(n):
				plt.cla()
				#plt.imshow(R[j,...,0], cmap='jet', interpolation='nearest')
				#print nlx_[j,...], nly_[j,...]
				plt.quiver(np.mean(nlx_[j,...,0]), np.mean(nly_[j,...,0]))
				plt.draw()
				raw_input(j)

def real_steer_comparison():
	"""Experiment to demonstrate the angular selectivity of the convolution"""
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	
	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_filters = 10
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None,28,28,1])
	v0 = tf.placeholder(tf.float32, [3,1,1])
	#y = equi_steer_conv_(x, v0)
	z = equi_real_conv(x, v0)

	# Initializing the variables
	init = tf.initialize_all_variables()
	
	X = mnist.train.next_batch(100)[0][1,:]
	X = np.reshape(X, [1,28,28,1])
	X_ = np.fliplr(X).T
	X = np.stack((X, X_))
	X = X.reshape([2,28,28,1])
	V0 = np.random.randn(3,1,1).astype(np.float32)
	#V0 = np.ones((3,1,1)).astype(np.float32)
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		X, Y = sess.run(z, feed_dict={x : X, v0 : V0})
	
	#R_0 = np.sqrt(np.sum(Y[0,...]**2, axis=2))
	#X_0, Y_0 = np.squeeze(Y[0,:,:,0])/R_0, np.squeeze(Y[0,:,:,1])/R_0
	X = np.squeeze(X)
	Y = np.squeeze(Y)
	R = np.sqrt(X**2 + Y**2)
	X, Y = X/R, Y/R
	
	plt.figure(1)
	plt.imshow(R[0], cmap='gray', interpolation='nearest')
	plt.quiver(X[0], Y[0])
	plt.figure(2)
	plt.imshow(R[1], cmap='jet', interpolation='nearest')
	plt.quiver(X[1], Y[1])
	plt.show()
	
def Z_steer_comparison():
	"""Experiment to demonstrate the angular selectivity of the convolution"""
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)	

	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_filters = 10
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None,28,28,1])
	v = tf.placeholder(tf.float32, [3,3,1,1])
	z = tf.nn.conv2d(x, v, strides=(1,1,1,1), padding='VALID')

	# Initializing the variables
	init = tf.initialize_all_variables()
	
	X = mnist.train.next_batch(100)[0][1,:]
	X = np.reshape(X, [1,28,28,1])
	X_ = np.fliplr(X).T
	X = np.stack((X, X_))
	X = X.reshape([2,28,28,1])
	V = np.random.randn(3,3,1,1).astype(np.float32)
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		X = sess.run(z, feed_dict={x : X, v : V})
	
	X = np.squeeze(X)
	X_T = np.flipud(X[1].T)
	
	plt.figure(1)
	plt.imshow(X[0], cmap='gray', interpolation='nearest')
	plt.figure(2)
	plt.imshow(X_T, cmap='gray', interpolation='nearest')
	plt.figure(3)
	plt.imshow(X[0] - X_T, cmap='gray', interpolation='nearest')
	plt.figure(4)
	plt.imshow(np.squeeze(V), cmap='gray', interpolation='nearest')
	plt.show()

def complex_steer_test():
	"""Test the complex convolutional filter"""
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	
	# Network Parameters
	n_input = 784 
	n_filters = 10
	
	# tf Graph input
	N = 50
	x = tf.placeholder(tf.float32, [N,28,28,1])
	v0 = tf.placeholder(tf.float32, [1,1,2*1,3])
	b0 = tf.placeholder(tf.float32, [3,])
	v1 = tf.placeholder(tf.float32, [1,1,2*3,1])
	
	y = equi_steer_conv(x, v0)
	#mp = complex_maxpool2d(y, k=2)	# For now process R independently everywhere
	y = complex_relu(y, b0)
	z = complex_steer_conv(y, v1)

	# Initializing the variables
	init = tf.initialize_all_variables()
	
	X = mnist.train.next_batch(100)[0][1,:]
	X = np.reshape(X, [28,28])
	X_ = []
	
	for i in xrange(N):
		angle = i*(360./N)
		X_.append(sciint.rotate(X, angle, reshape=False))
	X = np.reshape(np.stack(X_), [N,28,28,1])
	
	V0 = np.random.randn(1,1,2*1,3).astype(np.float32)
	B0 = np.random.randn(3).astype(np.float32)-0.5
	V1 = np.random.randn(1,1,2*3,1).astype(np.float32)
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		Z = sess.run(z, feed_dict={x : X, v0 : V0, b0 : B0, v1 : V1})
		
	Zx, Zy = Z
	R = np.sqrt(Zx**2 + Zy**2)
	Zx = np.squeeze(Zx/R)
	Zy = np.squeeze(Zy/R)
	R = np.squeeze(R)
	
	plt.figure(1)
	plt.ion()
	plt.show()
	for i in xrange(N):
		plt.cla()
		plt.imshow(R[i], cmap='jet', interpolation='nearest')
		plt.quiver(Zx[i], Zy[i])
		plt.show()
		raw_input(i*(360./N))
	
def complex_small_patch_test():
	"""Test the steer_conv on small rotated patches"""
	N = 50
	k=3
	X = np.random.randn(k**2)
	Q = get_Q(k=k)

	X = gen_data(X, N, Q)
	X = np.reshape(X, [N,3,3,1])
	
	x = tf.placeholder('float', [None,k,k,1], name='x')
	v0 = tf.placeholder('float', [3,1,1], name='v0')
	v1 = tf.placeholder('float', [1,1,2,1], name='v1')
	b0 = tf.placeholder('float', [1,], name='b0')
	esc1 = equi_real_conv(x, v0, order=1)
	esc1 = complex_relu(esc1, b0)
	z = equi_complex_conv(esc1, v1, k=1)
	
	V0 = np.random.randn(3,1,1)
	V1 = np.random.randn(1,1,2,1)
	B0 = np.random.rand(1)

	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		X, Y = sess.run(z, feed_dict={x : X, v0 : V0, v1 : V1, b0 : B0})
	
	R = np.sqrt(X**2 + Y**2)
	A = np.arctan2(Y, X)
	fig = plt.figure(1)
	theta = np.linspace(0, 2*np.pi, N)
	plt.plot(theta, np.squeeze(R), 'b')
	plt.plot(theta, np.squeeze(A), 'r')
	plt.show()

def small_patch_test():
	"""Test the steer_conv on small rotated patches"""
	N = 50
	X = np.random.randn(9)	#arange(9)
	Q = get_Q()
	X = gen_data(X, N, Q)
	X = np.reshape(X, [N,3,3,1])
	
	V = np.ones([3,1,1])
	#V[:,:,1,:] *= 20
	V = V/np.sqrt(np.sum(V**2))
	
	x = tf.placeholder('float', [None,3,3,1], name='x')
	v = tf.placeholder('float', [3,1,1], name='v')
	z = equi_real_conv(x, v)
	print V
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		X, Y = sess.run(z, feed_dict={x : X, v : V})
	
	R = np.sqrt(X**2 + Y**2)
	A = np.arctan2(Y, X)
	fig = plt.figure(1)
	theta = np.linspace(0, 2*np.pi, N)
	plt.plot(theta, np.squeeze(R), 'b')
	plt.plot(theta, np.squeeze(A), 'r')
	plt.show()
	

def gen_data(X, N, Q):
	# Get rotation
	theta = np.linspace(0, 2*np.pi, N)
	Y = []
	for t in theta:
		Y.append(reproject(Q,X,t))
	Y = np.vstack(Y)
	return Y

def get_Q(k=3,n=2):
	"""Return a tensor of steerable filter bases"""
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)
	x, y = np.meshgrid(lin, lin)
	gdx = gaussian_derivative(x, y, x)
	gdy = gaussian_derivative(x, y, y)
	G0 = np.reshape(gdx/np.sqrt(np.sum(gdx**2)), [k*k])
	G1 = np.reshape(gdy/np.sqrt(np.sum(gdx**2)), [k*k])
	return np.vstack([G0,G1])

def reproject(Q, X, angle):
	"""Reproject X through Q rotated by some amount"""
	# Represent in Q-space
	Y = np.dot(Q,X)
	# Rotate
	R = np.asarray([[np.cos(angle), np.sin(angle)],
					[-np.sin(angle), np.cos(angle)]])
	return np.dot(Q.T, np.dot(R,Y))
	
def dot_blade_test():
	v = tf.placeholder('float', [1,1,6,3], 'v')
	v_ = dot_blade_filter(v)

	V = np.random.randn(1,1,6,3)
	
	init = tf.initialize_all_variables()	
	with tf.Session() as sess:
		sess.run(init)
		V_ = sess.run(v_, feed_dict={v : V})

	V, V_ = V_
	print V
	print V_