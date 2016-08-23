'''The Matrix Lie Group Convolutional module'''

import os
import sys
import time

import numpy as np
import tensorflow as tf

def gConv(X, filter_size, n_filters, name=''):
    """Create a group convolutional module"""
    # Create variables
    k = filter_size
    n_channels = int(X.get_shape()[3])
    Q = get_weights([k,k,1,k*k], name=name+'_Q')
    V = get_weights([k*k,n_channels*n_filters], name=name+'_V')
    b = get_weights([n_filters], name=name+'_b')
    # Project input X to Q-space
    Xq = channelwise_conv2d(X, Q, strides=(1,1,1,1), padding="VALID")
    # Project V to Q-space: each col of Q is a filter transformation
    Vq = tf.matmul(tf.transpose(tf.reshape(Q, [k*k,k*k])), V)
    Vq = tf.reshape(Vq, [1,1,k*k,n_channels,n_filters])
    Vq = tf.transpose(Vq, perm=[0,3,1,4,2])
    # Get angle
    Xqsh = tf.shape(Xq)
    Xq = to_filter_patch_pairs(Xq, Xqsh)
    angle = get_signed_angle(Vq[:,:,:,:,:2],Xq[:,:,:,:,:2])
    angle = angle_to_image(angle, Xqsh)
    # Get response
    response = Xq
    return angle, Xq 

def get_rotation_as_vector(phi,k):
    """Return the Jordan block rotation matrix for the Lie Group"""
    R = []
    for i in xrange(np.floor((k*k)/2.)):
        R.append(tf.cos((i+1)*phi))
        R.append(tf.sin((i+1)*phi))
    if k % 2 == 1:
        R.append(1.)
    return tf.pack(R)
    
def channelwise_conv2d(X, Q, strides=(1,1,1,1), padding="VALID"):
    """Convolve X with Q on each channel independently.
    
    X: input tensor of shape [b,h,w,c]
    Q: rotation tensor of shape [h*w,h*w]
    
    returns: tensor of shape [b,c,h,w,m].
    """
    Xsh = tf.shape(X)
    X = tf.transpose(X, perm=[0,3,1,2])
    X = tf.reshape(X, tf.pack([Xsh[0]*Xsh[3],Xsh[1],Xsh[2],1]))
    Z = tf.nn.conv2d(X, Q, strides=strides, padding=padding)
    Zsh = tf.shape(Z)
    return tf.reshape(Z, tf.pack([Xsh[0],Xsh[3],Zsh[1],Zsh[2],Zsh[3]]))

def to_filter_patch_pairs(X, Xsh):
    '''Convert tensor [b,c,h,w,m] -> [b,c,hw,1,m]'''
    return tf.reshape(X, tf.pack([Xsh[0],Xsh[1],Xsh[2]*Xsh[3],1,Xsh[4]]))

def from_filter_patch_pairs(X, Xsh):
    '''Convert from filter-patch pairings'''
    return tf.reshape(X, tf.pack([Xsh[0],Xsh[1],Xsh[2],Xsh[3],Xsh[4]]))

def angle_to_image(X, Xsh):
    '''Convert from angular filter-patch pairings to standard image format'''
    return tf.reshape(X, tf.pack([Xsh[0],Xsh[2],Xsh[3],-1]))

def atan2(y,x):
    '''Compute the classic atan2 function between y and x'''
    arg1 = y / (tf.sqrt(tf.pow(y,2) + tf.pow(x,2)) + x)
    z1 = 2*tf.atan(arg1)
    
    arg2 = (tf.sqrt(tf.pow(y,2) + tf.pow(x,2)) - x) / y
    z2 = 2*tf.atan(arg2)
    
    z_ = tf.select(x>0,z1,z2)
    z = tf.select(tf.logical_and(tf.equal(y,0),(x<0)),np.pi*tf.ones_like(z_),z_)
    
    z = tf.select(tf.equal(y*x,0),tf.zeros_like(z_),z_)
    return z

def get_signed_angle(u,v):
    '''Get the signed angle from one vector to another'''
    # Reshape tensors
    ush = tf.shape(u)
    vsh = tf.shape(v)
    u = tf.tile(u,[vsh[0],1,vsh[2],1,1])
    v = tf.tile(v,[1,1,1,ush[3],1])
    # Compute angles
    dot = tf.reduce_sum(u*v,reduction_indices=[1,4])
    ext = v[:,:,:,:,1]*u[:,:,:,:,0] - v[:,:,:,:,0]*u[:,:,:,:,1]
    ext = tf.reduce_sum(ext, reduction_indices=[1])
    return atan2(ext, dot)

def modulus(x,y):
    '''Perform x % y and maintain sgn(x) = sgn(y)'''
    return x - y*tf.floordiv(x, y)

def get_weights(filter_shape, collection=None, name=''):
    W_init = tf.random_normal(filter_shape)
    return tf.Variable(W_init, collections=collection, name=name)

























