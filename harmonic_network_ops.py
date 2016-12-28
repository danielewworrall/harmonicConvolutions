"""
Core Harmonic Convolution Implementation

This file contains the core implementation of our harmonic convolutions.
We assume that:
	- the name_scopes these get called in is unique
	- tensors for the convolutional filters and biases have already been initialised
	  using get_weights_dict() and get_bias_dict() specified in harmonic_network_helpers.py.
	  Please see harmonic_network_models.py for examples of how this is done.
"""

import numpy as np
import tensorflow as tf


def conv(X, W, strides, padding, name):
	"""Shorthand for tf.nn.conv2d"""
	return tf.nn.conv2d(X, W, strides=strides, padding=padding, name=name)


def h_conv(X, Q, P=None, strides=(1,1,1,1), padding='VALID', filter_size=3,
				  max_order=1, name='N'):
	"""Inter-order (cross-stream) convolutions can be implemented as single
	convolutions. For this we store data as 6D tensors and filters as 8D
	tensors, at convolution, we reshape down to 4D tensors and expand again.
	
	X: tensor shape [mbatch,h,w,channels,complex,order]
	Q: tensor dict---reshaped to [h,w,in,in.comp,in.ord,out,out.comp,out.ord]
	P: tensor dict---phases
	strides: as per tf convention (default (1,1,1,1))
	padding: as per tf convention (default VALID)
	filter_size: (default 3)
	max_order: (default 1)
	name: (default N)
	"""
	with tf.name_scope('hconv'+str(name)) as scope:
		# Build data tensor
		Xsh = X.get_shape().as_list()
		X_ = tf.reshape(X, tf.concat(0,[Xsh[:3],[-1]]))
		
		# Build filter
		Q = get_filters(Q, filter_size=filter_size, P=P)
		#   Construct the stream-convolutions as one big filter
		Q_ = []
		for output_order in xrange(max_order+1):
			# For each output order build input
			Qr = []
			Qi = []
			for input_order in xrange(Xsh[3]):
				fo = output_order - input_order
				c = Q[np.abs(fo)]
				s = np.sign(fo)
				# Choose a different filter depending on whether imput is real
				if Xsh[4] == 2:
					Qr += [c[0],-s*c[1]]
					Qi += [c[1],s*c[0]]
				else:
					Qr += [c[0]]
					Qi += [c[1]]
			Q_ += [tf.concat(2, Qr), tf.concat(2, Qi)]
		Q_ = tf.concat(3, Q_)
		
		R = conv(X_, Q_, strides, padding, name+'cconv')
		Rsh = R.get_shape().as_list()
		ns = tf.concat(0, [Rsh[:3],[max_order+1,2],[Rsh[3]/(2*(max_order+1))]])
		return tf.reshape(R, ns)


##### NONLINEARITIES #####
def complex_nonlinearity(X, b, fnc, eps=1e-4):
	"""Apply the nonlinearity described by the function handle fnc: R -> R+ to
	the magnitude of X. CAVEAT: fnc must map to the non-negative reals R+.
	
	Output U + iV = fnc(R+b) * (A+iB)
	where  A + iB = Z/|Z|
	
	X: dict of channels {rotation order: (real, imaginary)}
	b: dict of biases {rotation order: real-valued bias}
	fnc: function handle for a nonlinearity. MUST map to non-negative reals R+
	eps: regularization since grad |Z| is infinite at zero (default 1e-4)
	"""
	b = tf.concat(3, [expand_ba(bv,5,0) for __, bv in b.iteritems()])
	magnitude = sum_magnitudes(X, eps)
	Rb = tf.add(magnitude, b)
	c = tf.div(fnc(Rb), magnitude)
	return c*X


def h_batch_norm(X, fnc, train_phase, decay=0.99, eps=1e-4,
					   name='complexBatchNorm', outerScope='complexBatchNormOuter', device='/cpu:0'):
	"""Batch normalization for the magnitudes of X
	
	X: dict of channels {rotation order: (real, imaginary)}
	fnc: function handle for a nonlinearity. MUST map to non-negative reals R+
	train_phase: boolean flag True: training mode, False: test mode
	decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
	eps: regularization since grad |Z| is infinite at zero (default 1e-4)
	name: (default complexBatchNorm)
	"""
	magnitude = sum_magnitudes(X, eps)
	Rb = batch_norm(magnitude, train_phase, decay=decay, name=name, device=device)
	c = tf.div(fnc(Rb), magnitude)
	return c*X


def batch_norm(X, train_phase, decay=0.99, name='batchNorm', device='/cpu:0'):
	"""Batch normalization module.
	
	X: tf tensor
	train_phase: boolean flag True: training mode, False: test mode
	decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
	name: (default batchNorm)
	
	Source: bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
	batch-normalization-in-tensorflow"""
	n_out = X.get_shape().as_list()[-3:]
	
	with tf.device(device):
		beta = tf.get_variable(name+'_beta', dtype=tf.float32, shape=n_out,
			initializer=tf.constant_initializer(0.0))
		gamma = tf.get_variable(name+'_gamma', dtype=tf.float32, shape=n_out,
			initializer=tf.constant_initializer(1.0))
		pop_mean = tf.get_variable(name+'_pop_mean', dtype=tf.float32, shape=n_out,
			trainable=False)
		pop_var = tf.get_variable(name+'_pop_var', dtype=tf.float32, shape=n_out,
			trainable=False)
		batch_mean, batch_var = tf.nn.moments(X, [0,1,2], name=name + 'moments')
	ema = tf.train.ExponentialMovingAverage(decay=decay)

	def mean_var_with_update():
		ema_apply_op = ema.apply([batch_mean, batch_var])
		pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
		pop_var_op = tf.assign(pop_var, ema.average(batch_var))

		with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
			return tf.identity(batch_mean), tf.identity(batch_var)

	mean, var = tf.cond(train_phase, mean_var_with_update,
				lambda: (pop_mean, pop_var))
	normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
	return normed


def sum_magnitudes_(X, eps=1e-4, keep_dims=True):
	"""Sum the magnitudes of each of the complex feature maps in X.
	
	Output U = sum_i |X_i|
	
	X: dict of channels {rotation order: (real, imaginary)}
	eps: regularization since grad |Z| is infinite at zero (default 1e-4)
	"""
	R = tf.reduce_sum(tf.square(X), reduction_indices=[4], keep_dims=keep_dims)
	return tf.sqrt(R + eps)


def sum_magnitudes(X, eps=1e-4, keep_dims=True):
	"""Experimental"""
	R = tf.reduce_sum(tf.square(X), reduction_indices=[4], keep_dims=keep_dims)
	return tf.sqrt(R + eps)



##### CREATING VARIABLES #####
def to_constant_float(Q):
	"""Converts a numpy tensor to a tf constant float
	
	Q: numpy tensor
	"""
	Q = tf.Variable(Q, trainable=False) 
	return tf.to_float(Q)


def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W', device='/cpu:0'):
	"""Initialize weights variable with He method
	
	filter_shape: list of filter dimensions
	W_init: numpy initial values (default None)
	std_mult: multiplier for weight standard deviation (default 0.4)
	name: (default W)
	device: (default /cpu:0)
	"""
	with tf.device(device):
		if W_init == None:
			stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:3]))
		return tf.get_variable(name, dtype=tf.float32, shape=filter_shape,
				initializer=tf.random_normal_initializer(stddev=stddev))


##### FUNCTIONS TO CONSTRUCT STEERABLE FILTERS #####
def get_filters(R, filter_size, P=None):
	"""Return a complex filter of the form $u(r,t,psi) = R(r)e^{im(t-psi)}"""
	filters = {}
	k = filter_size
	for m, r in R.iteritems():
		rsh = r.get_shape().as_list()
		# Get the basis matrices
		cosine, sine = get_complex_basis_matrices(k, order=m)
		cosine = tf.reshape(cosine, tf.pack([k*k, rsh[0]]))
		sine = tf.reshape(sine, tf.pack([k*k, rsh[0]]))
		
		# Project taps on to rotational basis
		r = tf.reshape(r, tf.pack([rsh[0],rsh[1]*rsh[2]]))
		ucos = tf.reshape(tf.matmul(cosine, r), tf.pack([k, k, rsh[1], rsh[2]]))
		usin = tf.reshape(tf.matmul(sine, r), tf.pack([k, k, rsh[1], rsh[2]]))
		
		if P is not None:
			# Rotate basis matrices
			ucos = tf.cos(P[m])*ucos + tf.sin(P[m])*usin
			usin = -tf.sin(P[m])*ucos + tf.cos(P[m])*usin
		filters[m] = (ucos, usin)
	return filters


def get_complex_basis_matrices(filter_size, order=1):
	"""Return complex basis component e^{imt} (ODD sizes only).
	
	filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
	order: rotation order (default 1)
	"""
	k = filter_size
	tap_length = int(((k+1)*(k+3))/8)
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)
	X, Y = np.meshgrid(lin, lin)
	R = np.sqrt(X**2 + Y**2)
	unique = np.unique(R)
	theta = np.arctan2(-Y, X)
	
	# There will be a cosine and quadrature sine mask
	cmasks = []
	smasks = []
	for i in xrange(tap_length):
		if order == 0:
			# For order == 0 there is nonzero weight on the center pixel
			cmask = (R == unique[i])*1.
			cmasks.append(to_constant_float(cmask))
			smask = (R == unique[i])*0.
			smasks.append(to_constant_float(smask))
		elif order > 0:
			# For order > 0 there is zero weights on the center pixel
			if unique[i] != 0.:
				cmask = (R == unique[i])*np.cos(order*theta)
				cmasks.append(to_constant_float(cmask))
				smask = (R == unique[i])*np.sin(order*theta)
				smasks.append(to_constant_float(smask))
	cmasks = tf.pack(cmasks, axis=-1)
	cmasks = tf.reshape(cmasks, [k,k,tap_length-(order>0)])
	smasks = tf.pack(smasks, axis=-1)
	smasks = tf.reshape(smasks, [k,k,tap_length-(order>0)])
	return cmasks, smasks


##### SPECIAL FUNCTIONS #####
def mean_pooling(X, ksize=(1,1,1,1), strides=(1,1,1,1)):
	"""Implement mean pooling on complex-valued feature maps. The complex mean
	on a local receptive field, is performed as mean(real) + i*mean(imag)
	
	X: dict of channels {rotation order: (real, imaginary)}
	ksize: kernel size 4-tuple (default (1,1,1,1))
	strides: stride size 4-tuple (default (1,1,1,1))
	"""
	"""
	Y = {}
	for k, v in X.iteritems():
		y0 = tf.nn.avg_pool(v[0], ksize=ksize, strides=strides,
							padding='VALID', name='mean_pooling')
		y1 = tf.nn.avg_pool(v[1], ksize=ksize, strides=strides,
							padding='VALID', name='mean_pooling')
		Y[k] = (y0,y1)
	"""
	Xsh = X.get_shape()
	# Collapse output the order, complex, and channel dimensions
	X_ = tf.reshape(X, tf.concat(0,[Xsh[:3],[-1]]))
	Y = tf.nn.avg_pool(X_, ksize=ksize, strides=strides, padding='VALID',
					   name='mean_pooling')
	Ysh = Y.get_shape()
	new_shape = tf.concat(0, [Ysh[:3],Xsh[3:]])
	return tf.reshape(Y, new_shape)


def get_complex_basis_functions(filter_size, order):
	"""Return complex exponential basis functions of order order
	
	filter_size: linear dimensions, must be an int
	order: the filter order, must be an int (pos and neg allowed)
	"""
	k = filter_size
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)
	X, Y = np.meshgrid(lin, lin)
	theta = np.arctan2(-Y, X)
	R = np.cos(order*theta)
	R = to_constant_float(R/(np.sum(R**2)+1e-6))
	I = np.sin(order*theta)
	I = to_constant_float(I/(np.sum(I**2)+1e-6))
	return tf.reshape(R, tf.pack([k,k,1,1])), tf.reshape(I, tf.pack([k,k,1,1]))


def expand_ba(value, n_before, n_after):
	"""Expand tensor with 1D n_before and n_after"""
	for __ in xrange(n_before):
		value = tf.expand_dims(value, 0)
	for __ in xrange(n_after):
		value = tf.expand_dims(value, -1)
	return value
