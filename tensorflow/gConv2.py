'''The Matrix Lie Group Convolutional module'''

import os
import sys
import time

import numpy as np
import scipy.linalg as scilin
import tensorflow as tf


def gConv(X, filter_size, n_filters, name=''):
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
    
def channelwise_conv2d(X, Q, strides=(1,1,1,1), padding="VALID"):
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

def cart_to_polar(X):
    """Input shape [m,:,:,:,:], output (r, theta). Assume d=9"""
    t = []
    r = []
    for i in xrange(4):
        t_ = atan2(X[2*i+1,:,:,:,:], X[2*i,:,:,:,:])
        r_ = tf.sqrt(tf.pow(X[2*i,:,:,:,:],2) + tf.pow(X[2*i+1,:,:,:,:],2))
        t.append(t_)
        r.append(r_)
    t.append(tf.zeros_like(t[3]))
    r.append(X[8,:,:,:,:])
    t = tf.pack(t)
    r = tf.pack(r)
    return (r,t)

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























