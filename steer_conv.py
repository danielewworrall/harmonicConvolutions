'''The Matrix Lie Group Convolutional module'''

import os
import sys
import time

import numpy as np
import scipy.linalg as scilin
import tensorflow as tf


def complex_conv(X, Q, strides=(1,1,1,1), padding='VALID', name='complexConv'):
    """Convolve a complex valued input X and complex-valued filter Q. Output is
    computed as (Xr + iXi)*(Qr + iQi) = (Xr*Qr - Xi*Qi) + i(Xr*Qi + Xi*Qr),
    where * denotes convolution.
    
    X: complex input stored as (real, imaginary)
    Q: complex filter stored as (real, imaginary)
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    name: (default complexConv)
    """
    Xr, Xi = X
    Qr, Qi = Q
    Rrr = tf.nn.conv2d(Xr, Qr, strides=strides, padding=padding, name='comprr')
    Rii = tf.nn.conv2d(Xi, Qi, strides=strides, padding=padding, name='compii')
    Rri = tf.nn.conv2d(Xr, Qi, strides=strides, padding=padding, name='compri')
    Rir = tf.nn.conv2d(Xi, Qr, strides=strides, padding=padding, name='compir')
    Rr = Rrr - Rii
    Ri = Rri + Rir
    return Rr, Ri

def real_input_conv(X, R, filter_size=3, strides=(1,1,1,1), padding='VALID',
                        name='riec'):
    """Equivariant complex convolution for a real input e.g. an image.
    
    X: tf tensor
    R: dict of filter coefficients {rotation order: (real, imaginary)}
    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    name: (default riec)
    
    Returns dict filter responses {order: (real, imaginary)}
    """
    Q = get_complex_filters(R, filter_size=filter_size)
    Z = {}
    for m, q in Q.iteritems():
        Zr = tf.nn.conv2d(X, q[0], strides=strides, padding=padding,
                          name='reic_real')
        Zi = tf.nn.conv2d(X, q[1], strides=strides, padding=padding,
                          name='reic_im')
        Z[m] = (Zr, Zi)
    return Z

def complex_input_conv(X, R, filter_size=3, output_orders=[0,],
                           strides=(1,1,1,1), padding='VALID', name='ciec'):
    """Equivariant complex convolution for a complex input e.g. feature maps.
    
    X: dict of channels {rotation order: (real, imaginary)}
    R: dict of filter coefficients {rotation order: (real, imaginary)}
    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    output_orders: list of rotation orders to output (default [0,])  
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    name: (default riec)
    
    Returns dict filter responses {order: (real, imaginary)}
    """
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
                              padding=padding)
            if order not in Z.keys():
                Z[order] = []
            Z[order].append(Z_)
    
    # Z is a dictionary of convolutional responses from each previous layer
    # feature map of rotation orders [A,B,...,C] to each feature map in this
    # layer of rotation orders [X,Y,...,Z]. At each map M in [X,Y,...,Z] we
    # sum the inputs from each F in [A,B,...,C].
    return sum_complex_tensor_list(Z)

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

def sum_complex_tensor_list(Z):
    """Z is a dict {order: [(real,im), (real,im), (real,im)]}. This function
    sums all the real parts and all the imaginary parts for each order. I think
    there is a better way to do this by representing each order as a single
    feature stack.
    """
    output = {}
    for order, response_list in Z.iteritems():
        reals = []
        ims = []
        for re, im in response_list:
            reals.append(re)
            ims.append(im)
        output[order] = (tf.add_n(reals), tf.add_n(ims))
    return output

##### NONLINEARITIES #####
def complex_relu(X, b, eps=1e-4):
    """Apply a ReLU to the modulus of the complex feature map.
    
    Output U + iV = ReLU(|Z| + b)*(A + iB)
    where  A + iB = Z/|Z|
    
    X: dict of channels {rotation order: (real, imaginary)}
    b: dict of biases {rotation order: real-valued bias}
    eps: regularization since grad |Z| is infinite at zero (default 1e-4)
    """
    R = {}
    for m, r in X.iteritems():
        magnitude = tf.sqrt(tf.square(r[0]) + tf.square(r[1]) + eps)
        Rb = tf.nn.bias_add(magnitude, b[m])
        c = tf.nn.relu(Rb)/magnitude
        R[m] = (r[0]*c, r[1]*c)
    return R

def complex_relu_of_sum(Z, b, eps=1e-4):
    """Apply a ReLU to the modulus of the sum of complex feature maps.
    
    Output U = complex_relu(sum(Z))
    
    X: dict of channels {rotation order: (real, imaginary)}
    b: dict of biases {rotation order: real-valued bias}
    eps: regularization since grad |Z| is infinite at zero (default 1e-4)
    """
    R = []
    for m, r in Z.iteritems():
        R_ = tf.sqrt(tf.square(r[0]) + tf.square(r[1]) + eps)
        R.append(tf.nn.bias_add(R_,b[m]))
    return tf.nn.relu(tf.add_n(R))

##### CREATING VARIABLES #####
def to_constant_float(Q):
    """Converts a numpy tensor to a tf constant float
    
    Q: numpy tensor
    """
    Q = tf.Variable(Q, trainable=False)
    return tf.to_float(Q)

def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W'):
    """Initialize weights variable with He method
    
    filter_shape: list of filter dimensions
    W_init: numpy initial values (default None)
    std_mult: multiplier for weight standard deviation (default 0.4)
    name: (default W)
    """
    if W_init == None:
        stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:2]))
        W_init = tf.random_normal(filter_shape, stddev=stddev)
    return tf.Variable(W_init, name=name)

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

def batch_norm(x, n_out, phase_train, name='bn'):
    """bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
    batch-normalization-in-tensorflow"""
    with tf.variable_scope(name) as scope:
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name=scope.name+'beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name=scope.name+'gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2],
                                              name=scope.name+'moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed




