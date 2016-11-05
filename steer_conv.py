'''The Matrix Lie Group Convolutional module'''

import os
import sys
import time

import numpy as np
import scipy.linalg as scilin
import tensorflow as tf

<<<<<<< HEAD
def complex_conv(X, Q, strides=(1,1,1,1), padding='VALID', name='N'):
	"""Convolve a complex valued input X and complex-valued filter Q. Output is
	computed as (Xr + iXi)*(Qr + iQi) = (Xr*Qr - Xi*Qi) + i(Xr*Qi + Xi*Qr),
	where * denotes convolution.
	
	X: complex input stored as (real, imaginary)
	Q: complex filter stored as (real, imaginary)
	strides: as per tf convention (default (1,1,1,1))
	padding: as per tf convention (default VALID)
	name: (default N)
	"""
	with tf.name_scope('complexConv'+name) as scope:
		Xr, Xi = X
		Qr, Qi = Q
		Rrr = tf.nn.conv2d(Xr, Qr, strides=strides, padding=padding, name='rr'+name)
		Rii = tf.nn.conv2d(Xi, Qi, strides=strides, padding=padding, name='ii'+name)
		Rri = tf.nn.conv2d(Xr, Qi, strides=strides, padding=padding, name='ri'+name)
		Rir = tf.nn.conv2d(Xi, Qr, strides=strides, padding=padding, name='ir'+name)
		Rr = Rrr - Rii
		Ri = Rri + Rir
		return Rr, Ri

def real_input_conv(X, R, filter_size=3, strides=(1,1,1,1), padding='VALID',
						name='N'):
	"""Equivariant complex convolution for a real input e.g. an image.
	
	X: tf tensor
	R: dict of filter coefficients {rotation order: (real, imaginary)}
	filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
	strides: as per tf convention (default (1,1,1,1))
	padding: as per tf convention (default VALID)
	name: (default N)
	
	Returns dict filter responses {order: (real, imaginary)}
	"""
	with tf.name_scope('reic'+str(name)) as scope:
		Q = get_complex_filters(R, filter_size=filter_size)
		Z = {}
		for m, q in Q.iteritems():
			Zr = tf.nn.conv2d(X, q[0], strides=strides, padding=padding,
							  name='reic_real'+name)
			Zi = tf.nn.conv2d(X, q[1], strides=strides, padding=padding,
							  name='reic_im'+name)
			Z[m] = (Zr, Zi)
		return Z

def complex_input_conv(X, R, filter_size=3, output_orders=[0,],
						   strides=(1,1,1,1), padding='VALID', name='N'):
	"""Equivariant complex convolution for a complex input e.g. feature maps.
	
	X: dict of channels {rotation order: (real, imaginary)}
	R: dict of filter coefficients {rotation order: (real, imaginary)}
	filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
	output_orders: list of rotation orders to output (default [0,])  
	strides: as per tf convention (default (1,1,1,1))
	padding: as per tf convention (default VALID)
	name: (default N)
	
	Returns dict filter responses {order: (real, imaginary)}
	"""
	with tf.name_scope('ceic'+str(name)) as scope:
		# Perform initial scan to link up all filter orders with input image orders.
		pairings = get_key_pairings(X, R, output_orders)
		Q = get_complex_filters(R, filter_size=filter_size)
		
		Z = {}
		for m, v in pairings.iteritems():
			for pair in v:
				q_, x_ = pair					   # filter key, input key
				order = q_ + x_
				s, q = np.sign(q_), Q[np.abs(q_)]   # key sign, filter
				x = X[x_]						   # input
				# For negative orders take conjugate of positive order filter.
				Z_ = complex_conv(x, (q[0], s*q[1]), strides=strides,
								  padding=padding, name=name)
				if order not in Z.keys():
					Z[order] = []
				Z[order].append(Z_)
		
		# Z is a dictionary of convolutional responses from each previous layer
		# feature map of rotation orders [A,B,...,C] to each feature map in this
		# layer of rotation orders [X,Y,...,Z]. At each map M in [X,Y,...,Z] we
		# sum the inputs from each F in [A,B,...,C].
		return sum_complex_tensor_dict(Z)

def real_input_rotated_conv(X, R, psi, filter_size=3, strides=(1,1,1,1), 
							padding='VALID', name='N'):
	"""Equivariant complex convolution for a real input e.g. an image.
	
	X: tf tensor
	R: dict of filter coefficients {rotation order: (real, imaginary)}
	psi: dict of filter phases {rotation order: phase}
	filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
	strides: as per tf convention (default (1,1,1,1))
	padding: as per tf convention (default VALID)
	name: (default N)
	
	Returns dict filter responses {order: (real, imaginary)}
	"""
	with tf.name_scope('reic'+str(name)) as scope:
		Q = get_complex_rotated_filters(R, psi, filter_size=filter_size)
		Z = {}
		for m, q in Q.iteritems():
			Zr = tf.nn.conv2d(X, q[0], strides=strides, padding=padding,
							  name='reic_real'+name)
			Zi = tf.nn.conv2d(X, q[1], strides=strides, padding=padding,
							  name='reic_im'+name)
			Z[m] = (Zr, Zi)
		return Z

def complex_input_rotated_conv(X, R, psi, filter_size=3, output_orders=[0,],
						   strides=(1,1,1,1), padding='VALID', name='N'):
	"""Equivariant complex convolution for a complex input e.g. feature maps.
	
	X: dict of channels {rotation order: (real, imaginary)}
	R: dict of filter coefficients {rotation order: (real, imaginary)}
	psi: dict of filter phases {rotation order: phase}
	filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
	output_orders: list of rotation orders to output (default [0,])  
	strides: as per tf convention (default (1,1,1,1))
	padding: as per tf convention (default VALID)
	name: (default N)
	
	Returns dict filter responses {order: (real, imaginary)}
	"""
	with tf.name_scope('ceic'+str(name)) as scope:
		# Perform initial scan to link up all filter orders with input image orders.
		pairings = get_key_pairings(X, R, output_orders)
		Q = get_complex_rotated_filters(R, psi, filter_size=filter_size)
		
		Z = {}
		for m, v in pairings.iteritems():
			for pair in v:
				q_, x_ = pair					   # filter key, input key
				order = q_ + x_
				s, q = np.sign(q_), Q[np.abs(q_)]   # key sign, filter
				x = X[x_]						   # input
				# For negative orders take conjugate of positive order filter.
				Z_ = complex_conv(x, (q[0], s*q[1]), strides=strides,
								  padding=padding, name=name)
				if order not in Z.keys():
					Z[order] = []
				Z[order].append(Z_)
		
		# Z is a dictionary of convolutional responses from each previous layer
		# feature map of rotation orders [A,B,...,C] to each feature map in this
		# layer of rotation orders [X,Y,...,Z]. At each map M in [X,Y,...,Z] we
		# sum the inputs from each F in [A,B,...,C].
		return sum_complex_tensor_dict(Z)
=======

def complex_conv(X, Q, strides=(1,1,1,1), padding='VALID', name='N'):
    """Convolve a complex valued input X and complex-valued filter Q. Output is
    computed as (Xr + iXi)*(Qr + iQi) = (Xr*Qr - Xi*Qi) + i(Xr*Qi + Xi*Qr),
    where * denotes convolution.
    
    X: complex input stored as (real, imaginary)
    Q: complex filter stored as (real, imaginary)
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    name: (default N)
    """
    with tf.name_scope('complexConv'+name) as scope:
        Xr, Xi = X
        Qr, Qi = Q
        Rrr = tf.nn.conv2d(Xr, Qr, strides=strides, padding=padding, name='rr'+name)
        Rii = tf.nn.conv2d(Xi, Qi, strides=strides, padding=padding, name='ii'+name)
        Rri = tf.nn.conv2d(Xr, Qi, strides=strides, padding=padding, name='ri'+name)
        Rir = tf.nn.conv2d(Xi, Qr, strides=strides, padding=padding, name='ir'+name)
        Rr = Rrr - Rii
        Ri = Rri + Rir
        return Rr, Ri

def real_input_conv(X, R, filter_size=3, strides=(1,1,1,1), padding='VALID',
                        name='N'):
    """Equivariant complex convolution for a real input e.g. an image.
    
    X: tf tensor
    R: dict of filter coefficients {rotation order: (real, imaginary)}
    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    name: (default N)
    
    Returns dict filter responses {order: (real, imaginary)}
    """
    with tf.name_scope('reic'+str(name)) as scope:
        Q = get_complex_filters(R, filter_size=filter_size)
        Z = {}
        for m, q in Q.iteritems():
            Zr = tf.nn.conv2d(X, q[0], strides=strides, padding=padding,
                              name='reic_real'+name)
            Zi = tf.nn.conv2d(X, q[1], strides=strides, padding=padding,
                              name='reic_im'+name)
            Z[m] = (Zr, Zi)
        return Z

def complex_input_conv(X, R, filter_size=3, output_orders=[0,],
                           strides=(1,1,1,1), padding='VALID', name='N'):
    """Equivariant complex convolution for a complex input e.g. feature maps.
    
    X: dict of channels {rotation order: (real, imaginary)}
    R: dict of filter coefficients {rotation order: (real, imaginary)}
    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    output_orders: list of rotation orders to output (default [0,])  
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    name: (default N)
    
    Returns dict filter responses {order: (real, imaginary)}
    """
    with tf.name_scope('ceic'+str(name)) as scope:
        # Perform initial scan to link up all filter orders with input image orders.
        pairings = get_key_pairings(X, R, output_orders)
        Q = get_complex_filters(R, filter_size=filter_size)
        
        Z = {}
        for m, v in pairings.iteritems():
            for pair in v:
                q_, x_ = pair                       # filter key, input key
                order = q_ + x_
                s, q = np.sign(q_), Q[np.abs(q_)]   # key sign, filter
                x = X[x_]                           # input
                # For negative orders take conjugate of positive order filter.
                Z_ = complex_conv(x, (q[0], s*q[1]), strides=strides,
                                  padding=padding, name=name)
                if order not in Z.keys():
                    Z[order] = []
                Z[order].append(Z_)
        
        # Z is a dictionary of convolutional responses from each previous layer
        # feature map of rotation orders [A,B,...,C] to each feature map in this
        # layer of rotation orders [X,Y,...,Z]. At each map M in [X,Y,...,Z] we
        # sum the inputs from each F in [A,B,...,C].
        return sum_complex_tensor_dict(Z)

def real_input_rotated_conv(X, R, psi, filter_size=3, strides=(1,1,1,1), 
                            padding='VALID', name='N'):
    """Equivariant complex convolution for a real input e.g. an image.
    
    X: tf tensor
    R: dict of filter coefficients {rotation order: (real, imaginary)}
    psi: dict of filter phases {rotation order: phase}
    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    name: (default N)
    
    Returns dict filter responses {order: (real, imaginary)}
    """
    with tf.name_scope('reic'+str(name)) as scope:
        Q = get_complex_rotated_filters(R, psi, filter_size=filter_size)
        Z = {}
        for m, q in Q.iteritems():
            Zr = tf.nn.conv2d(X, q[0], strides=strides, padding=padding,
                              name='reic_real'+name)
            Zi = tf.nn.conv2d(X, q[1], strides=strides, padding=padding,
                              name='reic_im'+name)
            Z[m] = (Zr, Zi)
        return Z

def complex_input_rotated_conv(X, R, psi, filter_size=3, output_orders=[0,],
                           strides=(1,1,1,1), padding='VALID', name='N'):
    """Equivariant complex convolution for a complex input e.g. feature maps.
    
    X: dict of channels {rotation order: (real, imaginary)}
    R: dict of filter coefficients {rotation order: (real, imaginary)}
    psi: dict of filter phases {rotation order: phase}
    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    output_orders: list of rotation orders to output (default [0,])  
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    name: (default N)
    
    Returns dict filter responses {order: (real, imaginary)}
    """
    with tf.name_scope('ceic'+str(name)) as scope:
        # Perform initial scan to link up all filter orders with input image orders.
        pairings = get_key_pairings(X, R, output_orders)
        Q = get_complex_rotated_filters(R, psi, filter_size=filter_size)
        
        Z = {}
        for m, v in pairings.iteritems():
            for pair in v:
                q_, x_ = pair                       # filter key, input key
                order = q_ + x_
                s, q = np.sign(q_), Q[np.abs(q_)]   # key sign, filter
                x = X[x_]                           # input
                # For negative orders take conjugate of positive order filter.
                Z_ = complex_conv(x, (q[0], s*q[1]), strides=strides,
                                  padding=padding, name=name)
                if order not in Z.keys():
                    Z[order] = []
                Z[order].append(Z_)
        
        # Z is a dictionary of convolutional responses from each previous layer
        # feature map of rotation orders [A,B,...,C] to each feature map in this
        # layer of rotation orders [X,Y,...,Z]. At each map M in [X,Y,...,Z] we
        # sum the inputs from each F in [A,B,...,C].
        return sum_complex_tensor_dict(Z)
>>>>>>> 2b4ea0d49e47df165f5e59a94a279a253cdf65f6

def get_key_pairings(X, R, output_orders):
	"""Finds combinations of all inputs and filters, such that
	input_order + filter_order = output_order
	
	X: dict of channels {rotation order: (real, imaginary)}
	R: dict of filter coefficients {rotation order: (real, imaginary)}
	output_orders: list of rotation orders to output 
	
	Returns {order : (r,x)} pairs.
	"""
	X_keys = np.asarray(X.keys())
	R_keys = np.asarray(mirror_filter_keys(R.keys()))[:,np.newaxis]
	# The compatibility matrix lists all sums of key pairings
	compatibility = X_keys + R_keys
	pairings = {}
	for order in output_orders:
		where = np.argwhere(compatibility == order)
		pairings[order] = []
		for k in where:
			pairings[order].append((R_keys[k[0],0], X_keys[k[1]]))
	return pairings

def mirror_filter_keys(R_keys):
	"""Add negative component to filter keys e.g. [0,1,2]->[-2,-1,0,1,2]
	
	R_keys: list of positive orders e.g. [0,1,2,...]
	"""
	new_keys = []
	for key in R_keys:
		if key == 0:
			new_keys.append(key)
		if key > 0:
			new_keys.append(key)
			new_keys.append(-key)
	return sorted(new_keys)

def sum_complex_tensor_dict(X):
	"""X is a dict of lists of tuples of complex numbers {order: [(real,im), \
	(real,im), ...]}. This function sums all the real parts and all the
	imaginary parts for each order. I think there is a better way to do this by
	representing each order as a single feature stack.
	
	X: dict of lists of complex tuples {order: [(real,im), (real,im), ...]}
	"""
	output = {}
	for order, response_list in X.iteritems():
		reals = []
		ims = []
		for re, im in response_list:
			reals.append(re)
			ims.append(im)
		output[order] = (tf.add_n(reals), tf.add_n(ims))
	return output

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
	R = {}
	for m, r in X.iteritems():
		magnitude = tf.sqrt(tf.square(r[0]) + tf.square(r[1]) + eps)
		Rb = tf.nn.bias_add(magnitude, b[m])
		c = fnc(Rb)/magnitude
		R[m] = (r[0]*c, r[1]*c)
	return R

def complex_batch_norm(X, fnc, phase_train, decay=0.99, eps=1e-4,
					   name='complexBatchNorm', outerScope='complexBatchNormOuter', device='/cpu:0'):
	"""Batch normalization for the magnitudes of X
	
	X: dict of channels {rotation order: (real, imaginary)}
	fnc: function handle for a nonlinearity. MUST map to non-negative reals R+
	phase_train: boolean flag True: training mode, False: test mode
	decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
	eps: regularization since grad |Z| is infinite at zero (default 1e-4)
	name: (default complexBatchNorm)
	"""
	#with tf.variable_scope(outerScope) as scope:
	R = {}
	idx = 0
	for m, r in X.iteritems():
		magnitude = tf.sqrt(tf.square(r[0]) + tf.square(r[1]) + eps)
		Rb = batch_norm(magnitude, phase_train, decay=decay, name=name+'_'+str(idx), device=device)
		c = fnc(Rb)/magnitude
		R[m] = (r[0]*c, r[1]*c)
		idx += 1
	return R

def batch_norm(X, phase_train, decay=0.99, name='batchNorm', device='/cpu:0'):
	"""Batch normalization module.
	
	X: tf tensor
	phase_train: boolean flag True: training mode, False: test mode
	decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
	name: (default batchNorm)
	
	Source: bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
	batch-normalization-in-tensorflow"""
	n_out = X.get_shape().as_list()[-1]
	
	with tf.device(device):
		beta = tf.get_variable(name+'_beta', dtype=tf.float32, shape=[n_out],
			initializer=tf.constant_initializer(0.0))
		gamma = tf.get_variable(name+'_gamma', dtype=tf.float32, shape=[n_out],
			initializer=tf.constant_initializer(1.0))
	batch_mean, batch_var = tf.nn.moments(X, [0,1,2],
										  name=name + 'moments')
	ema = tf.train.ExponentialMovingAverage(decay=decay)

	def mean_var_with_update():
		ema_apply_op = ema.apply([batch_mean, batch_var])
		with tf.control_dependencies([ema_apply_op]):
			return tf.identity(batch_mean), tf.identity(batch_var)

	mean, var = tf.cond(phase_train, mean_var_with_update,
				lambda: (ema.average(batch_mean), ema.average(batch_var)))
	normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
	return normed

def sum_magnitudes(X, eps=1e-4):
	"""Sum the magnitudes of each of the complex feature maps in X.
	
	Output U = sum_i |X_i|
	
	X: dict of channels {rotation order: (real, imaginary)}
	eps: regularization since grad |Z| is infinite at zero (default 1e-4)
	"""
	R = []
	for m, r in X.iteritems():
		R.append(tf.sqrt(tf.square(r[0]) + tf.square(r[1]) + eps))
	return tf.add_n(R)

def stack_magnitudes(X, eps=1e-4):
    """Stack the magnitudes of each of the complex feature maps in X.
    
    Output U = tf.concat(|X_0|, |X_1|, ...)
    
    X: dict of channels {rotation order: (real, imaginary)}
    eps: regularization since grad |Z| is infinite at zero (default 1e-4)
    """
    R = []
    for m, r in X.iteritems():
        R.append(tf.sqrt(tf.square(r[0]) + tf.square(r[1]) + eps))
    return tf.concat(3, R, name='concat')

##### CREATING VARIABLES #####
def to_constant_float(Q):
<<<<<<< HEAD
	"""Converts a numpy tensor to a tf constant float
	
	Q: numpy tensor
	"""
	#Q = tf.Variable(Q, trainable=False) ############################################
	return tf.to_float(Q)

def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W', device='/cpu:0'):
	"""Initialize weights variable with He method
	
	filter_shape: list of filter dimensions
	W_init: numpy initial values (default None)
	std_mult: multiplier for weight standard deviation (default 0.4)
	name: (default W)
	"""
	with tf.device(device):
		if W_init == None:
			stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:3]))
		return tf.get_variable(name, dtype=tf.float32, shape=filter_shape,
				initializer=tf.random_normal_initializer(stddev=stddev))
=======
    """Converts a numpy tensor to a tf constant float
    
    Q: numpy tensor
    """
    #Q = tf.Variable(Q, trainable=False)###########################################
    return tf.to_float(Q)

def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W'):
    """Initialize weights variable with He method
    
    filter_shape: list of filter dimensions
    W_init: numpy initial values (default None)
    std_mult: multiplier for weight standard deviation (default 0.4)
    name: (default W)
    """
    if W_init == None:
        stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:3]))
        W_init = tf.random_normal(filter_shape, stddev=stddev)
        #W_init = tf.random_uniform(filter_shape, maxval=np.sqrt(12.)*stddev/4.)
    return tf.Variable(W_init, name=name)
>>>>>>> 2b4ea0d49e47df165f5e59a94a279a253cdf65f6

##### FUNCTIONS TO CONSTRUCT STEERABLE FILTERS #####
def get_complex_filters(R, filter_size):
	"""Return a complex filter of the form u(r,t) = R(r)e^{imt}, using a
	Cartesian representation so u(x,y) = X + iY, where m is rotation order
	and t is rotation angle clockwise about the origin.
	
	R: dict of filter coefficients {rotation order: (real, imaginary)}
	filter_size: int of filter height/width (default 3) CAVEAT: ODD supported 
	"""
	filters = {}
	k = filter_size
	for m, r in R.iteritems():
		rsh = r.get_shape().as_list()
		cosine, sine = get_complex_basis_matrices(k, order=m)
		cosine = tf.reshape(cosine, tf.pack([k*k, rsh[0]]))
		sine = tf.reshape(sine, tf.pack([k*k, rsh[0]]))
		# Project taps on to rotational basis
		r = tf.reshape(r, tf.pack([rsh[0],rsh[1]*rsh[2]]))
		ucos = tf.reshape(tf.matmul(cosine, r), tf.pack([k, k, rsh[1], rsh[2]]))
		usin = tf.reshape(tf.matmul(sine, r), tf.pack([k, k, rsh[1], rsh[2]]))
		filters[m] = (ucos, usin)
	return filters

def get_complex_rotated_filters(R, psi, filter_size):
	"""Return a complex filter of the form $u(r,t,psi) = R(r)e^{im(t-psi)}"""
	filters = {}
	k = filter_size
	for m, r in R.iteritems():
		rsh = r.get_shape().as_list()
		# Get the basis matrices
		cmasks, smasks = get_complex_basis_matrices(filter_size, order=m)
		# Reshape and project taps on to basis
		cosine = tf.reshape(cmasks, tf.pack([k*k, rsh[0]]))
		sine = tf.reshape(smasks, tf.pack([k*k, rsh[0]]))
		# Project taps on to rotational basis
		r = tf.reshape(r, tf.pack([rsh[0],rsh[1]*rsh[2]]))
		ucos = tf.reshape(tf.matmul(cosine, r), tf.pack([k, k, rsh[1], rsh[2]]))
		usin = tf.reshape(tf.matmul(sine, r), tf.pack([k, k, rsh[1], rsh[2]]))
		print ucos.get_shape(), psi[m].get_shape()
		# Rotate basis matrices
		cosine = tf.cos(psi[m])*ucos + tf.sin(psi[m])*usin
		sine = -tf.sin(psi[m])*ucos + tf.cos(psi[m])*usin
		filters[m] = (cosine, sine)
	return filters

def get_complex_rotated_filters(R, psi, filter_size):
    """Return a complex filter of the form $u(r,t,psi) = R(r)e^{im(t-psi)}"""
    filters = {}
    k = filter_size
    for m, r in R.iteritems():
        rsh = r.get_shape().as_list()
        # Get the basis matrices
        cmasks, smasks = get_complex_basis_matrices(filter_size, order=m)
        # Reshape and project taps on to basis
        cosine = tf.reshape(cmasks, tf.pack([k*k, rsh[0]]))
        sine = tf.reshape(smasks, tf.pack([k*k, rsh[0]]))
        # Project taps on to rotational basis
        r = tf.reshape(r, tf.pack([rsh[0],rsh[1]*rsh[2]]))
        ucos = tf.reshape(tf.matmul(cosine, r), tf.pack([k, k, rsh[1], rsh[2]]))
        usin = tf.reshape(tf.matmul(sine, r), tf.pack([k, k, rsh[1], rsh[2]]))
        # Rotate basis matrices
        cosine = tf.cos(psi[m])*ucos + tf.sin(psi[m])*usin
        sine = -tf.sin(psi[m])*ucos + tf.cos(psi[m])*usin
        filters[m] = (cosine, sine)
    return filters

def get_complex_basis_matrices(filter_size, order=1):
<<<<<<< HEAD
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
	Y = {}
	for k, v in X.iteritems():
		y0 = tf.nn.avg_pool(v[0], ksize=ksize, strides=strides,
							padding='VALID', name='mean_pooling')
		y1 = tf.nn.avg_pool(v[1], ksize=ksize, strides=strides,
							padding='VALID', name='mean_pooling')
		Y[k] = (y0,y1)
	return Y

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

def hough_module(X, weights, filter_size):
	"""Implement a differential Hough module, which is a variation of the
	equivariant voting module from Liu et al. (2012)
	
	X: dict of channels {rotation order: (real, imaginary)}
	weights: dict of univariate complex weights {rotation order: (real, imaginary)}
	filter_size: linear dimensions, must be an int
	"""
	# Create a basis spanning all the orders of the output feature maps
	response = {}
	Zr = 0
	Zi = 0
	for order, data in X.iteritems():
		Wr, Wi = weights[order]
		# May need to include an envelope term
		basis = get_complex_basis_functions(filter_size, -order)
		Yr, Yi = complex_conv(data, basis, padding='SAME', name='hough_conv')
		# May need to constrain W to lie on the circle
		Zr += Wr*Yr - Wi*Yi
		Zi += Wr*Yi + Wi*Yr
	return Zr, Zi




























=======
    """Return complex basis component e^{imt} (ODD sizes only).
    
    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    order: rotation order (default 1)
    """
    k = filter_size
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    X, Y = np.meshgrid(lin, lin)
    R = np.sqrt(X**2 + Y**2)
    unique = np.unique(R)
    theta = np.arctan2(-Y, X)
    tap_length = unique.shape[0]
    
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
>>>>>>> 2b4ea0d49e47df165f5e59a94a279a253cdf65f6

##### SPECIAL FUNCTIONS #####
def mean_pooling(X, ksize=(1,1,1,1), strides=(1,1,1,1)):
    """Implement mean pooling on complex-valued feature maps"""
    Y = {}
    for k, v in X.iteritems():
        y0 = tf.nn.avg_pool(v[0], ksize=ksize, strides=strides,
                            padding='VALID', name='mean_pooling')
        y1 = tf.nn.avg_pool(v[1], ksize=ksize, strides=strides,
                            padding='VALID', name='mean_pooling')
        Y[k] = (y0,y1)
    return Y
