'''The Matrix Lie Group Convolutional module'''

import os
import sys
import time

import numpy as np
import scipy.linalg as scilin
import tensorflow as tf


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

def steer_conv(X, V, b=None, strides=(1,1,1,1), padding='VALID', k=3, n=2,
               name='steerConv'):
    Q = get_basis(k=k,n=n)
    Z = channelwise_conv2d(X, Q, strides=strides, padding=padding, name=name)
    # 1d convolution to combine filters
    Y = tf.nn.conv2d(Z, V, strides=(1,1,1,1), padding='VALID', name=name+'1d')
    if b is not None:
        Y = tf.nn.bias_add(Y, b)
    return Y

def equi_steer_conv(X, V, strides=(1,1,1,1), padding='VALID', k=3, n=2,
                    name='equisteerConv'):
    """Steerable convolution returning max and argmax"""
    Xsh = tf.shape(X)
    Q = get_basis(k=k,n=n)
    
    Z = channelwise_conv2d(X, Q, strides=strides, padding=padding, name=name)
    V_dot, V_blade = dot_blade_filter(V) 
    Y_dot = tf.nn.conv2d(Z, V_dot, strides=(1,1,1,1), padding='VALID', name=name+'1d')
    Y_blade = tf.nn.conv2d(Z, V_blade, strides=(1,1,1,1), padding='VALID', name=name+'1d')
    return Y_dot, Y_blade

def equi_steer_conv_(X, V, strides=(1,1,1,1), padding='VALID', k=3, n=2,
                    name='equisteerConv'):
    """Steerable convolution returning max and argmax"""
    Q = get_basis(k=k,n=n)
    
    Qsh = Q.get_shape().as_list()
    Xsh = X.get_shape().as_list()
    tile_shape = tf.pack([1,1,Xsh[3],1])
    Q = tf.tile(Q, tile_shape, name='Q_tile')
    V, V_ = dot_blade_filter(V)
    
    V_ = tf.concat(3, [V,V_])
    Y = tf.nn.separable_conv2d(X, Q, V_, strides=strides, padding='VALID',
                               name='sep_conv')
    return Y

def complex_steer_conv(Z, V, strides=(1,1,1,1), padding='VALID', k=3, n=2,
                       name='complexsteerconv'):
    """Simpler complex steerable filter returning max real phase and modulus"""
    Zsh = Z[0].get_shape().as_list()
    tile_shape = tf.pack([1,1,Zsh[3],1])
    Q = get_complex_basis(k=k, n=n, wrap=0.)
    Q = (tf.tile(Q[0], tile_shape), tf.tile(Q[1], tile_shape))
    # Filter channels
    Y = complex_depthwise_conv(Z, Q, strides=strides, padding=padding, name='cd')
    # Filter dot blade
    return complex_dot_blade(Y, V)

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

def get_arg(Y):
    """Get the argument of the steerable convolution"""
    return atan2(Y[:,:,:,1,:],Y[:,:,:,0,:])

def get_modulus(Y):
    """Get the per-pixel modulus of a steerable convolution"""
    return tf.sqrt(tf.reduce_sum(tf.pow(Y,2.), reduction_indices=3))

def to_dot_blade(Y):
    """Convert tensor to dot-blade form for modulus/argument extraction"""
    Ysh = Y.get_shape().as_list()
    return tf.reshape(Y, tf.pack([-1,Ysh[1],Ysh[2],2,Ysh[3]/2]))

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

def interleave(A,B):
    """Interleave two 4D tensors along final axis"""
    Ash = A.get_shape().as_list()
    C = tf.pack([A,B], axis=-1)
    return tf.reshape(C, tf.pack([Ash[0],Ash[1],Ash[2],Ash[3]*2]))

def steer_pool(X):
    """Return the max and argmax over steering rotation"""
    R = tf.sqrt(tf.reduce_sum(tf.pow(X, 2.), reduction_indices=4))
    T = atan2(X[:,:,:,:,1],X[:,:,:,:,0])
    return R, T

def complex_maxpool2d_(X, k=2):
    """Max pool over complex valued feature maps by modulus only"""
    U, V = X
    R = tf.square(U) + tf.square(V)
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

def complex_maxpool2d(X, k=2):
    """Max pool over complex valued feature maps by modulus only"""
    U, V = X
    R = tf.square(U) + tf.square(V)
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

def complex_relu(Z, b, eps=1e-4):
    """Apply a ReLU to the modulus of the complex feature map"""
    X, Y = Z
    R2 = tf.square(X) + tf.square(Y)
    Rb = tf.nn.bias_add(R2,b)
    c = tf.nn.relu(Rb)/tf.maximum(R2,eps) #*tf.sign(Rb)
    X = X * c
    Y = Y * c
    return X, Y

def complex_relu_(Z, b, eps=1e-6):
    """Apply a ReLU to the modulus of the complex feature map"""
    X, Y = Z
    R = tf.sqrt(tf.square(X) + tf.square(Y))
    R = tf.maximum(R,eps)
    X = X/R
    Y = Y/R
    c = tf.nn.relu(tf.nn.bias_add(R,b), name='relu')
    X = c*X
    Y = c*Y
    return X, Y

def get_basis(k=3, n=2):
    """Return a tensor of steerable filter bases"""
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    x, y = np.meshgrid(lin, lin)
    gdx = gaussian_derivative(x, y, x)
    gdy = gaussian_derivative(x, y, y)
    G0 = np.reshape(gdx/np.sqrt(np.sum(gdx**2)), [k,k,1,1])
    G1 = np.reshape(gdy/np.sqrt(np.sum(gdy**2)), [k,k,1,1])
    return to_constant_variable(np.concatenate([G0,G1], axis=3))

def get_basis(k=3, n=2):
    """Return a learnable steerable basis"""
    pass

def get_complex_basis(k=3, n=2, wrap=1.):
    """Return a tensor of complex steerable filter bases (X, Y)"""
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    X, Y = np.meshgrid(lin, lin)
    Y = np.flipud(Y)
    X = tf.to_float(X)
    Y = tf.to_float(Y)
    R = tf.sqrt(X**2 + Y**2)
    modulus = R*tf.exp(-R**2)
    phase = wrap*atan2(Y, X) 
    Rcos = tf.reshape(modulus*tf.cos(phase), [k,k,1,1])
    Rsin = tf.reshape(modulus*tf.sin(phase), [k,k,1,1])
    X0, Y0 = Rcos, Rsin
    X1, Y1 = -Rsin, Rcos
    X = tf.concat(3, [X0,X1])
    Y = tf.concat(3, [Y0,Y1])
    return (X,Y)

def to_constant_variable(Q):
    Q = tf.Variable(Q, trainable=False)
    return tf.to_float(Q)

def gaussian_derivative(x,y,direction):
    return -2*direction*np.exp(-(x**2 + y**2))

def gConv_(X, filter_size, n_filters, name=''):
    """Create a group convolutional module"""
    # Create variables
    k = filter_size
    n_channels = int(X.get_shape()[3])
    print('N_channels: %i' % (n_channels,))
    print('N_filters: %i' % (n_filters,))
    Q = get_weights([k,k,1,k*k], W_init=Q_init(), name=name+'_Q')
    V = get_weights([k*k,n_channels*n_filters], name=name+'_V')         # [h*w,c*f]
    # Project input X to Q-space
    Xq = channelwise_conv2d(X, Q, strides=(1,1,1,1), padding="VALID")   # [m,c,b,h',w']
    # Project V to Q-space: each col of Q is a filter transformation
    Q_ = tf.transpose(tf.reshape(Q, [k*k,k*k]))
    Vq = tf.matmul(Q_, V)
    
    Vq = tf.reshape(Vq, [1,k*k,n_channels,n_filters])                   # [1,m,c,f]
    Vq = tf.transpose(Vq, perm=[1,2,0,3])                               # [m,c,1,f]
    # Get angle
    Xqsh = tf.shape(Xq)                                                 # [m,c,b,h',w']
    Xq = to_filter_patch_pairs(Xq, Xqsh)                                # [m,c,bh'w',1]
    Vq, Xq = mutual_tile(Vq, Xq)    # Do we need a sanity check on this?# [m,c,bh'w',f]
    dot, ext = dot_ext_transform(Xq,Vq)                                 # [d,bh'w',f] [d,bh'w',f]
    angle = get_angle(dot[0,:,:], ext[0,:,:])                           # [bh'w',f]
    angle = tf.zeros_like(angle)
    # Get response
    response = get_response(angle, k, dot, ext, n_harmonics=4)
    # Reshape to image-like shape
    angle = fp_to_image(angle, Xqsh)                                    # [b,h',w',f]
    response = fp_to_image(response, Xqsh)                              # [b,h',w',f]
    return angle, response, V

def orthogonalize(Q):
    """Orthogonalize square Q"""
    Q = tf.reshape(Q, [9,9])
    S, U, V = tf.svd(Q, compute_uv=True, full_matrices=True)
    return tf.reshape(tf.matmul(U,tf.transpose(V)), [3,3,1,9])

def get_response(angle, k, dot, ext, n_harmonics=4):
    """Return the rotation response for the Lie Group up to n harmonics"""
    # Get response
    Rcos, Rsin = get_rotation_as_vectors(angle, k, n_harmonics=n_harmonics) # [d,bh'w',f]
    cos_response = tf.reduce_sum(dot*Rcos, reduction_indices=[0])       # [bh'w',f]
    sin_response = tf.reduce_sum(ext*Rsin, reduction_indices=[0])       # [bh'w',f]
    return cos_response + sin_response                                  # [bh'w',f]

def get_rotation_as_vectors(phi,k,n_harmonics=4):
    """Return the Jordan block rotation matrix for the Lie Group"""
    Rcos = []
    Rsin = []
    j = 1.
    for i in xrange(np.floor((k*k)/2.).astype(int)):
        if i >= n_harmonics:
            j = 0.
        Rcos.append(j*tf.cos((i+1)*phi))
        Rsin.append(j*tf.sin((i+1)*phi))
    if k % 2 == 1:
        Rcos.append(tf.ones_like(Rcos[-1]))
        Rsin.append(tf.zeros_like(Rsin[-1]))
    return tf.pack(Rcos), tf.pack(Rsin)

def channelwise_conv2d(X, Q, strides=(1,1,1,1), padding="VALID", name='conv'):
    """Convolve X with Q on each channel independently. Returns: tensor of
    shape [b,h',w',m*c].
    """
    Qsh = Q.get_shape().as_list()
    Xsh = X.get_shape().as_list()
    tile_shape = tf.pack([1,1,Xsh[3],1])
    Q = tf.tile(Q, tile_shape, name='Q_tile')
    Y = tf.nn.depthwise_conv2d(X, Q, strides=strides, padding=padding,
                               name=name+'chan_conv')
    return Y    
    
def channelwise_conv2d_(X, Q, strides=(1,1,1,1), padding="VALID"):
    """Convolve X with Q on each channel independently.
    
    X: input tensor of shape [b,h,w,c]
    Q: orthogonal tensor of shape [hw,hw]. Note h = w, m = hw
    
    returns: tensor of shape [m,c,b,h',w'].
    """
    Xsh = tf.shape(X)                                           # [b,h,w,c]
    X = tf.transpose(X, perm=[0,3,1,2])                         # [b,c,h,w]
    X = tf.reshape(X, tf.pack([Xsh[0]*Xsh[3],Xsh[1],Xsh[2],1])) # [bc,h,w,1]
    Z = tf.nn.conv2d(X, Q, strides=strides, padding=padding)    # [bc,h',w',m]
    Zsh = tf.shape(Z)
    Z = tf.reshape(Z, tf.pack([Xsh[0],Xsh[3],Zsh[1],Zsh[2],Zsh[3]])) # [b,c,h',w',m]
    return tf.transpose(Z, perm=[4,1,0,2,3])                    # [m,c,b,h',w']

def channelwise_conv2d_(X, Q, strides=(1,1,1,1), padding="VALID"):
    """Convolve X with Q on each channel independently. Using depthwise conv
    
    X: input tensor of shape [b,h,w,c]
    Q: orthogonal tensor of shape [hw,hw]. Note h = w, m = hw
    
    returns: tensor of shape [m,c,b,h',w'].
    """
    Xsh = tf.shape(X)
    Xsh_ = X.get_shape().as_list()
    Q_ = tf.tile(Q, [1,1,Xsh_[3],1])                             # [k,k,c,m]
    Z = tf.nn.depthwise_conv2d(X, Q_, strides=strides, padding=padding) # [b,h',w',c*k*k]
    Zsh = tf.shape(Z)
    Z_ = tf.reshape(Z, tf.pack([Xsh[0],Zsh[1],Zsh[2],Xsh[3],Zsh[3]/Xsh_[3]])) # [b,h',w',c,m]
    return tf.transpose(Z_, perm=[4,3,0,1,2])                    # [m,c,b,h',w']

def to_filter_patch_pairs(X, Xsh):
    """Convert tensor [m,c,b,h,w] -> [m,c,bhw,1]"""
    return tf.reshape(X, tf.pack([Xsh[0],Xsh[1],Xsh[2]*Xsh[3]*Xsh[4],1]))

def from_filter_patch_pairs(X, Xsh):
    """Convert from filter-patch pairings"""
    return tf.reshape(X, tf.pack([Xsh[0],Xsh[1],Xsh[2],Xsh[3],Xsh[4]]))

def fp_to_image(X, Xsh):
    """Convert from angular filter-patch pairings to standard image format"""
    return tf.reshape(X, tf.pack([Xsh[2],Xsh[3],Xsh[4],-1]))

def cart_to_polar(X, Y):
    """Input shape [m,:,:,:,:], output (r, theta). Assume d=9"""
    R = tf.sqrt(tf.pow(X,2.) + tf.pow(Y,2.))
    T = atan2(Y, X)
    return (R, T)

def polar_to_cart(R,T):
    """Polar to cartesian coordinates"""
    X = R*tf.cos(T)
    Y = R*tf.sin(T)
    return (X, Y)

def atan2(y, x, reg=1e-6):
    """Compute the classic atan2 function between y and x"""
    x = safe_reg(x)
    y = safe_reg(y)
    
    arg1 = y / (tf.sqrt(tf.pow(y,2) + tf.pow(x,2)) + x)
    z1 = 2*tf.atan(arg1)
    
    arg2 = (tf.sqrt(tf.pow(y,2) + tf.pow(x,2)) - x) / y
    z2 = 2*tf.atan(arg2)
    
    return tf.select(x>0,z1,z2)

def safe_reg(x, reg=1e-6):
    """Return the x, such that |x| >= reg"""
    return (2.*tf.to_float(tf.greater(x,0.))-1.)*(tf.abs(x) + reg)

def get_angle(dot, ext):
    """Get the angle in [0,2*pi] from one vector to another"""
    # Compute angles
    return modulus(atan2(ext, dot), 2*np.pi)

def dot_ext_transform(U,V):
    """Convert {U,V} to the dot-ext domain (vector representation of SO(N))"""
    # Dot: input [m,c,bh'w',f], [m,c,bh'w',f]
    dot = tf.reduce_sum(U*V,reduction_indices=[1])  # [m,bh'w',f]
    dotsh = tf.to_int32(tf.shape(dot)[0])
    seg_indices = tf.range(dotsh)/2
    dot = tf.segment_sum(dot, seg_indices)          # [ceil(m/2),bh'w',f]
    # Ext
    Vsh = tf.shape(V)
    V = tf.reshape(V, [Vsh[0],Vsh[1]*Vsh[2]*Vsh[3]])# [m,cbh'w'f]   
    V = tf.reshape(tf.matmul(blade_matrix(9),V), [Vsh[0],Vsh[1],Vsh[2],Vsh[3]]) # [m,c,bh'w',f]
    ext = tf.reduce_sum(U*V, reduction_indices=[1]) # [m,bh'w',f]
    return dot, tf.segment_sum(ext, seg_indices)    # [ceil(m/2),bh'w',f] [ceil(m/2),bh'w',f]

def blade_matrix(k):
    """Build the blade product matrix of order k"""
    blade = np.zeros([k,k])
    blade[k-1,k-1] = 1
    for i in xrange(int(np.floor(k/2.))):
        blade[(2*i)+1,2*i] = 1
        blade[2*i,(2*i)+1] = -1
    return tf.to_float(tf.identity(blade))
        
def mutual_tile(u,v):
    """Tile u and v to be the same shape"""
    ush = tf.shape(u)
    vsh = tf.shape(v)
    maxsh = tf.maximum(ush,vsh)
    u = tf.tile(u, maxsh/ush)
    v = tf.tile(v, maxsh/vsh)
    return u, v

def modulus(x,y):
    """Perform x % y and maintain sgn(x) = sgn(y)"""
    return x - y*tf.floordiv(x, y)

def get_weights(filter_shape, W_init=None, collection=None, name=''):
    if W_init == None:
        W_init = tf.random_normal(filter_shape, stddev=0.01)
    return tf.Variable(W_init, collections=collection, name=name)

def Q_init():
    Q = getQ()
    P = permuation_matrix()
    Q_ = np.real(np.dot(Q,P))
    return np.reshape(Q_, [3,3,1,9]).astype(np.float32)

def getQ():
    n = 9
    Q = np.eye(n, dtype=np.complex)
    Q[:n-1,:n-1] = scilin.dft(n-1)/(np.sqrt(n-1.))
    P = permuteFourier(Q)
    u = np.asarray([[1,1],[1j,-1j]])
    U = np.eye(n, dtype=np.complex)
    U[2:4,2:4] = u
    U[4:6,4:6] = u
    U[6:8,6:8] = u
    Q = np.real(np.dot(U,P))
    return Q

def permuteFourier(F):
    P = np.zeros((9,9))
    P[0,0] = 1
    P[1,4] = 1
    P[2,1] = 1
    P[3,7] = 1
    P[4,2] = 1
    P[5,6] = 1
    P[6,3] = 1
    P[7,5] = 1
    P[8,8] = 1
    return np.dot(P, F)

def permuation_matrix():
    P = np.zeros((9,9))
    P[0,0] = 1
    P[1,1] = 1
    P[2,2] = 1
    P[3,5] = 1
    P[4,8] = 1
    P[5,7] = 1
    P[6,6] = 1
    P[7,3] = 1
    P[8,4] = 1
    return P























