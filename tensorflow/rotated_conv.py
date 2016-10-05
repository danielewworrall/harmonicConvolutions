'''The Matrix Lie Group Convolutional module'''

import os
import sys
import time

import numpy as np
import scipy.linalg as scilin
import tensorflow as tf


def to_constant_variable(Q):
    """Converts a numpy tensor to a tf constant"""
    Q = tf.Variable(Q, trainable=False)
    return tf.to_float(Q)

def get_weights(filter_shape, W_init=None, collection=None, name=''):
    """Return normally initialized weight variables"""
    if W_init == None:
        W_init = tf.random_normal(filter_shape, stddev=0.01)
    return tf.Variable(W_init, collections=collection, name=name)

def equi_real_conv(X, V, strides=(1,1,1,1), padding='VALID', k=3, order=1,
                    name='equiRealConv'):
    """Equivariant real convolution. Returns a list of filter responses
    output = [Z0, Z1c, Z1s, Z2c, Z2s, ...]. Z0 is zeroth frequency, Z1 is
    the first frequency, Z2 the second etc., Z1c is the cosine response, Z1s
    is the sine response."""
    Xsh = tf.shape(X)
    Q = get_steerable_filter(V, order)
    Z = tf.nn.conv2d(X, Q, strides=strides, padding=padding, name='equireal')
    return tf.split(3, 2*order+1, Z)

def stack_moduli(Z, eps=1e-3):
    """Stack the moduli of the filter responses. Z is the output of a
    real_equi_conv."""
    R = []
    R.append(Z[0])
    for i in xrange(len(Z)/2):
        R.append(tf.sqrt(tf.square(Z[2*i+1]) + tf.square(Z[2*i+2]) + eps))
    return tf.concat(3, R)

def sum_moduli(Z, eps=1e-3):
    """Stack the moduli of the filter responses. Z is the output of a
    real_equi_conv."""
    R = []
    R.append(Z[0])
    for i in xrange(len(Z)/2):
        R.append(tf.sqrt(tf.square(Z[2*i+1]) + tf.square(Z[2*i+2]) + eps))
    return tf.add_n(R)

##### NONLINEARITIES #####
# Just use the phase_invariant_relu for now
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

def complex_relu(Z, b, order=1, eps=1e-4):
    """Apply a ReLU to the modulus of the complex feature map"""
    Z_ = []
    oddness = len(Z) % 2
    if oddness:
        Z_.append(tf.nn.bias_add(Z[0], b[0]))
    for i in xrange(order):
        X = Z[2*i-1-oddness]
        Y = Z[2*i-oddness]
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

##### FUNCTIONS TO CONSTRUCT STEERABLE FILTERS#####
def get_steerable_filter(V, order=1):
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
    for i in xrange(order):
        ord_ = str(i+1)
        Vi = tf.reshape(V[i+1], [Vsh[0]-1,Vsh[1]*Vsh[2]])
        Wx = tf.matmul(masks[i+1], Vi, name='Wx')
        W.append(tf.reshape(Wx, [k,k,Vsh[1],Vsh[2]]))
        W.append(-tf.reverse(tf.transpose(W[-1], perm=[1,0,2,3]), [False,True,False,False]))
    return tf.concat(3, W)

def get_basis_matrices(k, order=1):
    """Return tf cosine masks for custom tap learning (works with odd sizes)"""
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
            masks.append(to_constant_variable(mask))
        elif order > 0:
            if unique[i] != 0.:
                mask = (R == unique[i])*np.cos(order*theta)
                masks.append(to_constant_variable(mask))
    masks = tf.pack(masks, axis=-1)
    return tf.reshape(masks, [k,k,tap_length-(order>0)])


##### COMPLEX-VALUED STUFF---IGNORE FOR NOW#####
def equi_complex_conv(Z, V, strides=(1,1,1,1), padding='VALID', k=3, order=0,
                      name='equiComplexConv'):
    Zsh = Z[0].get_shape().as_list()
    tile_shape = tf.pack([1,1,Zsh[3],1])
    wrap = 0.
    Q = get_complex_basis(k=k, order=order, wrap=wrap)
    
    Q = (tf.tile(Q[0], tile_shape), -tf.tile(Q[1], tile_shape))
    # Filter channels
    Y = complex_depthwise_conv(Z, Q, strides=strides, padding=padding, name='cd')
    # Filter dot blade
    return complex_dot_blade(Y, V)

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

def get_steerable_complex_filter(V, order=0):
    """Return a steerable filter of order n from the input V"""
    Vsh = V.get_shape().as_list()     # [tap_length,i,o]
    k = int(np.sqrt(1 + 8.*Vsh[0]) - 2)
    masks = tf.reshape(get_basis_matrices(k, order=order), [k*k,Vsh[0]])
    
    V = tf.reshape(V, [Vsh[0],Vsh[1]*Vsh[2]])
    Wr = tf.matmul(masks, V, name='Wx')
    return tf.reshape(Wr, [k,k,Vsh[1],Vsh[2]])

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
        masks.append(to_constant_variable(mask))
    return masks

def complex_GAP(Z, W):
    """Take the average of a set of feature maps over the spatial dimensions
    and return the real part of a fully-connected complex transformation"""
    X, Y = Z
    R, I = W
    X = tf.reduce_mean(X, reduction_indices=[1,2])
    Y = tf.reduce_mean(Y, reduction_indices=[1,2])
    return tf.matmul(X, R) + tf.matmul(Y, I)

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








