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
	
####### Taco Cohen's stuff---I think#######
def gConv(X, V, b=None, strides=(1,1,1,1), padding='VALID', name='gConv'):
    """Run a Taco Cohen G-convolution module"""
    V_ = get_rotation_stack(V, name=name+'stack')
    VX = tf.nn.conv2d(X, V_, strides=strides, padding=padding, name=name+'2d')
    if b is not None:
        VXsh = tf.shape(VX)
        VX = tf.reshape(VX, [VXsh[0],VXsh[1],VXsh[2],4,VXsh[3]/4])
        VX = tf.reshape(tf.nn.bias_add(VX, b), [VXsh[0],VXsh[1],VXsh[2],VXsh[3]])
    return VX

def coset_pooling(X):
    # Coset pooling is max-pooling over local subgroups
    Xsh = tf.shape(X)
    X = tf.reshape(X, [Xsh[0],Xsh[1]*Xsh[2],4,Xsh[3]/4])
    X = tf.nn.max_pool(X, ksize=[1,1,4,1], strides=[1,1,4,1], padding='SAME')
    return tf.reshape(X, [Xsh[0],Xsh[1],Xsh[2],Xsh[3]/4])

def get_rotation_stack(V, name='rot_stack_concat'):
    """Return stack of 90 degree rotated V"""
    V0 = V
    V1 = rotate_90_clockwise(V0)
    V2 = rotate_90_clockwise(V1)
    V3 = rotate_90_clockwise(V2)
    return tf.concat(3, [V0,V1,V2,V3], name=name)

def rotate_90_clockwise(V):
    """Rotate a square filter clockwise by 90 degrees"""
    Vsh = V.get_shape().as_list()
    V = tf.reshape(V, [Vsh[0],Vsh[1],Vsh[2]*Vsh[3]])
    V_ = tf.image.rot90(V, k=1)
    return tf.reshape(V_, [Vsh[0],Vsh[1],Vsh[2],Vsh[3]])

##### Not working yet #####
# Two attempts at maxpooling. The problem is that the gradients don't work
def complex_maxpool2d_(X, k=2, eps=1e-3):
    """Max pool over complex valued feature maps by modulus only"""
    U, V = X
    R = tf.square(U) + tf.square(V) + eps
    max_, argmax = tf.nn.max_pool_with_argmax(R, [1,k,k,1], strides=[1,k,k,1],
                                              padding='VALID', name='cpool')
    Ush = tf.shape(U)
    batch_correct = tf.to_int64(tf.reduce_prod(Ush[1:])*tf.range(Ush[0]))
    argmax = argmax + tf.reshape(batch_correct, [Ush[0],1,1,1])
    
    U_flat = tf.reshape(U, [-1])
    V_flat = tf.reshape(V, [-1])
    
    U_ = tf.gather(U_flat, argmax)
    V_ = tf.gather(V_flat, argmax)
    
    return U_, V_

def complex_maxpool2d(X, k=2, eps=1e-3):
    """Max pool over complex valued feature maps by modulus only"""
    U, V = X
    R = tf.square(U) + tf.square(V) + eps
    max_, argmax = tf.nn.max_pool_with_argmax(R, [1,k,k,1], strides=[1,k,k,1],
                                              padding='VALID', name='cpool')
    argmax_list = tf.unpack(argmax, name='amunpack')
    U_list = tf.unpack(U, name='Uunpack')
    V_list = tf.unpack(V, name='Vunpack')
    U_ = []
    V_ = []
    for am, u, v in zip(argmax_list, U_list, V_list):
        u = tf.reshape(u, [-1])
        v = tf.reshape(v, [-1])
        U_.append(tf.gather(u, am))
        V_.append(tf.gather(v, am))
    U_ = tf.pack(U_)
    V_ = tf.pack(V_)
    
    return U_, V_

##### STEPHAN CAN IGNORE THIS#####
def complex_dot_blade(Z, V, name='complexdotblade'):
    V_dot, V_blade = dot_blade_filter(V)
    Dx = tf.nn.conv2d(Z, V_dot, strides=(1,1,1,1), padding='VALID', name='Dx')
    Bx = tf.nn.conv2d(Z, V_blade, strides=(1,1,1,1), padding='VALID', name='Bx')
    return (Dx, Bx)

def complex_depthwise_conv(Z, W, strides=(1,1,1,1), padding='VALID',
                             name='complexchannelwiseconv'):
    """
    Channelwise convolution using complex filters using a cartesian
    representation. Input tensors X = A+iB and filters W=U+iV. Returns: tensors
    AU+BV + i(AV-BV) of shape [b,h',w',m*c].
    """
    X, Y = Z
    U, V = W
    XU = tf.nn.depthwise_conv2d(X, U, strides=strides, padding=padding, name='XU')
    YV = tf.nn.depthwise_conv2d(Y, V, strides=strides, padding=padding, name='YV')
    return XU - YV

def dot_blade_filter(V):
    """Convert the [1,1,2i,o] filter to a dot-blade format size [1,1,2i,2o]"""
    V_ = tf.squeeze(V, squeeze_dims=[0,1])
    Vsh = V_.get_shape()
    V_ = tf.reshape(tf.transpose(V_), tf.pack([(Vsh[0]*Vsh[1])/2,2]))
    S = to_constant_variable(np.asarray([[0.,1.],[-1.,0.]]))
    V_blade = tf.matmul(V_,S)
    V_blade = tf.reshape(V_blade, tf.pack([1,1,Vsh[1],Vsh[0]]))
    V_blade = tf.transpose(V_blade, perm=[0,1,3,2])
    return V, V_blade

def equi_real_conv(X, V, strides=(1,1,1,1), padding='VALID', k=3, order=1,
                    name='equiRealConv'):
    """Rotationally equivariant real convolution. Returns a list of filter responses
    output = [Z0, Z1c, Z1s, Z2c, Z2s, ...]. Z0 is zeroth frequency, Z1 is
    the first frequency, Z2 the second etc., Z1c is the cosine response, Z1s
    is the sine response."""
    Q = get_steerable_real_filter(V, order=order)
    Z = tf.nn.conv2d(X, Q, strides=strides, padding=padding, name='equireal')
    return tf.split(3, 2*(order+1), Z)

def stack_moduli(Z, eps=1e-3):
    """Stack the moduli of the filter responses. Z is the output of a
    real_equi_conv."""
    R = []
    R.append(Z[0])
    for i in xrange(len(Z)/2):
        R.append(tf.sqrt(tf.square(Z[2*i+1]) + tf.square(Z[2*i+2]) + eps))
    return tf.concat(3, R)

def stack_moduli_dict(Z, eps=1e-3):
    """Stack the moduli of the filter responses from a dict Z, which is the
    output of a complex_symmetric_conv.
    """
    R = []
    for k, v in Z.iteritems():
        R.append(tf.sqrt(tf.square(v[0]) + tf.square(v[1]) + eps))
    return tf.concat(3, R)

def sum_moduli(Z, eps=1e-4):
    """Sum the moduli of the filter responses. Z is the output of a
    real_equi_conv."""
    R = []
    for i in xrange((len(Z)/2)):
        R.append(tf.sqrt(tf.square(Z[2*i]) + tf.square(Z[2*i+1]) + eps))
    return tf.add_n(R)

def sum_moduli_dict(Z, eps=1e-3):
    """Sum the moduli of the filter responses. Z is the output of a
    complex_symmetric_conv."""
    R = []
    for m, v in Z.iteritems():
        R.append(tf.sqrt(tf.square(v[0]) + tf.square(v[1]) + eps))
    return tf.add_n(R)

def phase_invariant_relu(Z, b, order=1, eps=1e-3):
    """Apply a ReLU to the modulus of the complex feature map, returning the
    modulus (which is phase invariant) only. Z and b should be lists of feature
    maps from the output of a real_equi_conv"""
    Z_ = []
    oddness = len(Z) % 2
    if oddness:
        Z_.append(tf.nn.bias_add(Z[0], b[0]))
    for i in xrange(order):
        X = Z[2*i-1-oddness]
        Y = Z[2*i-oddness]
        R = tf.sqrt(tf.square(X) + tf.square(Y) + eps)
        Rb = tf.nn.bias_add(R, b[i+1])
        Z_.append(tf.nn.relu(Rb))
    return Z_

def complex_relu(Z, b, eps=1e-4):
    """Apply a ReLU to the modulus of the complex feature map"""
    Z_ = []
    Z_.append(tf.nn.bias_add(Z[0], b[0]))
    for i in xrange(len(Z)/2):
        X = Z[2*i+1]
        Y = Z[2*i+2]
        R = tf.sqrt(tf.square(X) + tf.square(Y) + eps)
        Rb = tf.nn.bias_add(R, b[i+1])
        c = tf.nn.relu(Rb)/R
        Z_.append(X * c)
        Z_.append(Y * c)
    return Z_

def complex_softplus(Z, b, order=1, eps=1e-4):
    """Apply a ReLU to the modulus of the complex feature map"""
    Z_ = []
    oddness = len(Z) % 2
    if oddness:
        Z_.append(Z[0])
    for i in xrange(order):
        X = Z[2*i-1-oddness]
        Y = Z[2*i-oddness]
        R = tf.sqrt(tf.square(X) + tf.square(Y) + eps)
        Rb = tf.nn.bias_add(R,b)
        c = tf.nn.softplus(Rb)/R
        Z_.append(X * c)
        Z_.append(Y * c)
    return Z_

def get_steerable_real_filter(V, order=1):
    """Return a steerable filter up to frequency 'order' from the input V"""
    # Some shape maths
    Vsh = V[0].get_shape().as_list()     # [tap_length,i,o]
    k = int(np.sqrt(1 + 8.*Vsh[0]) - 2)
    
    # Generate the sinusoidal masks for steering
    masks = {}
    for i in xrange(order + 1):
        masks[i] = tf.reshape(get_basis_matrices(k, order=i), [k*k,Vsh[0]-(i>0)])
    
    # Build the filters from linear combinations of the sinusoid mask and the
    # radial weighting coefficients
    W = []
    V0 = tf.reshape(V[0], [Vsh[0],Vsh[1]*Vsh[2]])
    W.append(tf.reshape(tf.matmul(masks[0], V0, name='Wx'), [k,k,Vsh[1],Vsh[2]]))
    W.append(tf.zeros_like(W[-1]))
    for i in xrange(order):
        ord_ = str(i+1)
        Vi = tf.reshape(V[i+1], [Vsh[0]-1,Vsh[1]*Vsh[2]])
        Wx = tf.matmul(masks[i+1], Vi, name='Wx')
        W.append(tf.reshape(Wx, [k,k,Vsh[1],Vsh[2]]))
        W.append(-tf.reverse(tf.transpose(W[-1], perm=[1,0,2,3]), [False,True,False,False]))
    return tf.concat(3, W)

def get_steerable_filter(V, orders=[0]):
    """Return a steerable filter UP TO frequency 'order' from the input V"""
    # Some shape maths
    Vsh = V[0].get_shape().as_list()     # [tap_length,i,o]
    k = int(np.sqrt(1 + 8.*Vsh[0]) - 2)
    
    # Generate the sinusoidal masks for steering
    masks = {}
    for order in orders:
        bases = get_complex_basis_matrices(k, order=order)
        base_shape = tf.pack([k*k,V[order].get_shape()[0]])
        baseR = tf.reshape(bases[0], base_shape)
        baseI = tf.reshape(bases[1], base_shape)
        masks[order] = (baseR, baseI)
    # Build the filters from linear combinations of the sinusoid mask and the
    # radial weighting coefficients
    W = []
    for order in orders:
        Vi = V[order]
        Vish = Vi.get_shape()
        Vi = tf.reshape(Vi, tf.pack([Vish[0],Vish[1]*Vish[2]]))
        Wr = tf.matmul(masks[order][0], Vi, name='WR'+'_'+str(order))
        Wi = tf.matmul(masks[order][1], Vi, name='WI'+'_'+str(order))
        W.append(tf.reshape(Wr, tf.pack([k,k,Vish[1],Vish[2]])))
        W.append(tf.reshape(Wi, tf.pack([k,k,Vish[1],Vish[2]])))
    return tf.concat(3, W)

def get_steerable_complex_filter(V, order=0):
    """Return an order 0 complex steerable filter from the input V"""
    Vsh = V.get_shape().as_list()     # [tap_length,i,o]
    k = int(np.sqrt(1 + 8.*Vsh[0]) - 2)
    masks = tf.reshape(get_basis_matrices(k, order=order), [k*k,Vsh[0]])
    
    V = tf.reshape(V, [Vsh[0],Vsh[1]*Vsh[2]])
    Wr = tf.matmul(masks, V, name='Wx')
    Wr = tf.reshape(Wr, [k,k,Vsh[1],Vsh[2]])
    Wi = tf.reverse(tf.transpose(Wr, perm=[1,0,2,3]), [False, True, False, False])
    return Wr, Wi

def stack_responses(X):
    Z_ = []
    X_ = []
    Y_ = []
    Z_.append(X[0])
    for i in xrange(len(X)/2):
        X_.append(X[2*i+1])
        Y_.append(X[2*i+2])
    Z_.append(tf.concat(3, X_))
    Z_.append(tf.concat(3, Y_))
    return Z_

def complex_steer_conv(Z, V, strides=(1,1,1,1), padding='VALID', k=3, order=1,
                       name='complexsteerconv'):
    """Simpler complex steerable filter returning max real phase and modulus.
    Ignore this currently, it doesn't work"""
    Zsh = Z[0].get_shape().as_list()
    tile_shape = tf.pack([1,1,Zsh[3],1])
    wrap = 0.
    Q = get_complex_basis(k=k, order=order, wrap=wrap)
    
    Q = (tf.tile(Q[0], tile_shape), tf.tile(Q[1], tile_shape))
    # Filter channels
    Y = complex_depthwise_conv(Z, Q, strides=strides, padding=padding, name='cd')
    # Filter dot blade
    return complex_dot_blade(Y, V)

def get_complex_basis(k=3, order=2, wrap=1.):
    """Return a tensor of complex steerable filter bases (X, Y)"""
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    X, Y = np.meshgrid(lin, lin)
    Y = -Y
    X = tf.to_float(X)
    Y = tf.to_float(Y)
    
    tap_length = int(((k+1)*(k+3))/8)
    tap = get_weights([tap_length], name='tap')
    masks = get_complex_basis_masks(k)
    new_masks = []
    for i in xrange(tap_length):
        new_masks.append(masks[i]*tap[i])
    modulus = tf.add_n(new_masks, name='modulus')
    
    phase = wrap*atan2(Y, X) 
    Rcos = tf.reshape(modulus*tf.cos(phase), [k,k,1,1])
    Rsin = tf.reshape(modulus*tf.sin(phase), [k,k,1,1])
    X0, Y0 = Rcos, Rsin
    X1, Y1 = -Rsin, Rcos
    X = tf.concat(3, [X0,X1])
    Y = tf.concat(3, [Y0,Y1])
    return (X,Y)

def get_complex_basis_masks(k):
    """Return tf cosine masks for custom tap learning (works with odd sizes)"""
    tap_length = int(((k+1)*(k+3))/8)
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    X, Y = np.meshgrid(lin, lin)
    R = X**2 + Y**2
    unique = np.unique(R)
    
    masks = []
    for i in xrange(tap_length):
        mask = (R == unique[i])
        masks.append(to_constant_float(mask))
    return masks

def complex_GAP(Z, W):
    """Take the average of a set of feature maps over the spatial dimensions
    and return the real part of a fully-connected complex transformation"""
    X, Y = Z
    R, I = W
    X = tf.reduce_mean(X, reduction_indices=[1,2])
    Y = tf.reduce_mean(Y, reduction_indices=[1,2])
    return tf.matmul(X, R) + tf.matmul(Y, I)

def get_basis_matrices(k, order=1):
    """Return tf cosine masks for custom tap learning (works with odd sizes).
    k is filter size, order is rotation order"""
    tap_length = int(((k+1)*(k+3))/8)
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    X, Y = np.meshgrid(lin, lin)
    R = np.sqrt(X**2 + Y**2)
    unique = np.unique(R)
    theta = np.arctan2(-Y, X)

    masks = []
    for i in xrange(tap_length):
        if order == 0:
            mask = (R == unique[i])*1.
            masks.append(to_constant_float(mask))
        elif order > 0:
            if unique[i] != 0.:
                mask = (R == unique[i])*np.cos(order*theta)
                masks.append(to_constant_float(mask))
    masks = tf.pack(masks, axis=-1)
    return tf.reshape(masks, [k,k,tap_length-(order>0)])
