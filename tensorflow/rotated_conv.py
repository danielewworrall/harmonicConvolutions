'''The Matrix Lie Group Convolutional module'''

import os
import sys
import time

import numpy as np
import scipy.linalg as scilin
import tensorflow as tf


def real_input_equi_conv(X, R, filter_size=3, strides=(1,1,1,1), padding='VALID',
                        name='riec'):
    """Equivariant complex convolution for a real input e.g. an image.
    
    X: tf tensor
    R: dict of filter coefficients {rotation order: (real, imaginary)}
    filter_size: int of filter height/width (default 3)
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

def complex_symmetric_conv(X, R, filter_size=3, output_orders=[0,],
                           strides=(1,1,1,1), padding='VALID', name='ciec'):
    """Equivariant complex convolution for a complex input e.g. feature maps.
    
    X: dict of channels {rotation order: (real, imaginary)}
    R: dict of filter coefficients {rotation order: (real, imaginary)}
    filter_size: int of filter height/width (default 3)
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
            Z_ = complex_conv(x, (q[0], s*q[1]), strides=strides, padding=padding)
            if order not in Z.keys():
                Z[order] = []
            Z[order].append(Z_)
    return sum_complex_tensor_list(Z)

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
    """Add negative component to filter keys e.g. [0,1,2]->[-2,-1,0,1,2]"""
    new_keys = []
    for key in R_keys:
        if key == 0:
            new_keys.append(key)
        if key > 0:
            new_keys.append(key)
            new_keys.append(-key)
    return sorted(new_keys)

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
    """Converts a numpy tensor to a tf constant float"""
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

def get_complex_filters(R, filter_size):
    """Return a complex filter of the form u(r,t) = R(r)e^{imt}, but in #
    Cartesian coordinates. m is the rotation order, t the orientation and
    r the radius. R is a dict of filter taps with keys denoting the rotation
    orders.
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

##### COMPLEX CONVOLUTIONS #####
def get_complex_basis_matrices(k, order=1):
    """Return e^{i.order.t} masks for custom tap learning (works with odd sizes).
    k is filter size, order is rotation order"""
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

def complex_conv(Z, Q, strides=(1,1,1,1), padding='VALID', name='equiComplexConv'):
    Zr, Zi = Z
    Qr, Qi = Q
    Rrr = tf.nn.conv2d(Zr, Qr, strides=strides, padding=padding, name='comprr')
    Rii = tf.nn.conv2d(Zi, Qi, strides=strides, padding=padding, name='compii')
    Rri = tf.nn.conv2d(Zr, Qi, strides=strides, padding=padding, name='compri')
    Rir = tf.nn.conv2d(Zi, Qr, strides=strides, padding=padding, name='compir')
    Rr = Rrr - Rii
    Ri = Rri + Rir
    return Rr, Ri

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

def splice_responses(X, Z):
    """Splice the responses from X and Z"""
    print X
    print Z

##### COMPLEX-VALUED STUFF---IGNORE FOR NOW#####

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

def batch_norm(x, n_out, phase_train, name='bn'):
    """bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
    batch-normalization-in-tensorflow"""
    with tf.variable_scope(name) as scope:
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name=scope.name+'beta',
                           trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name=scope.name+'gamma',
                            trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name=scope.name+'moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed




