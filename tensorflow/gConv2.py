'''The Matrix Lie Group Convolutional module'''

import os
import sys
import time

import numpy as np
import tensorflow as tf

def gConv(X, filter_shape, name=''):
    """Create a group convolutional module"""
    Q = get_weights(Q_shape, collections='Q', name=name+'Q')
    V = get_weights(V_shape, collections='V', name=name+'V')
    b = get_weights(b_shape, collections='b', name=name+'b')
    # Project input X to Q-space
    Xq = channelwise_conv2d(X, Q, strides=(1,1,1,1), padding="VALID")
    # Project V to Q-space
    Vq = tf.matmul(tf.transpose(tf.reshape(Q, [9,9])), V)
    Vq = tf.reshape(Vq, [9,None])
    # Get angle
    tf.matmul(QV)

def gConv(X, Q, W, eps=1e-6):
    # Get L2-norms of subvectors---ordering of segments is arbitrary
    normQx = tf.sqrt(tf.segment_sum(tf.pow(Qx,2), [0,0,1,1,2,2,3,3,4]))
    normQw = tf.sqrt(tf.segment_sum(tf.pow(Qw,2), [0,0,1,1,2,2,3,3,4]))
    normQ = normQx * normQw
    # Elementwise multiply Qw and Qx along output axis of channelwise conv2d
    wQtQx = Qx * Qw
    dotSum = tf.segment_sum(wQtQx, [0,0,1,1,2,2,3,3,4])
    # Find the subvector angles for the rotations---eps is for regularization
    normDotSum = tf.truediv(dotSum, normQ + eps)
    # normDot is a tensor of dotProducts, we can return the angle using acos
    return tf.transpose(normDotSum, perm=[1,2,3,4,0])
    
def channelwise_conv2d(X, W, strides=(1,1,1,1), padding="VALID"):
    """Convolve _X with _W on each channel independently. The input _X will be a 
    tensor of shape [b,h,w,c], so reshape to [b*c,h,w,1], then apply conv2d. The
    result is a tensor of shape [b*c,h,w,m], we then reshape to [m,b,h,w,c].
    """
    Xsh = tf.shape(X)
    X = tf.transpose(X, perm=[0,3,1,2])
    X = tf.reshape(X, tf.pack([Xsh[0]*Xsh[3],Xsh[1],Xsh[2],1]))
    Z = tf.nn.conv2d(X, W, strides=strides, padding=padding)
    Zsh = tf.shape(Z)
    Z = tf.reshape(Z, tf.pack([Xsh[0],Xsh[3],Zsh[1],Zsh[2],Zsh[3]]))
    return tf.transpose(Z, perm=[4,0,2,3,1])

def atan2(y,x):
    '''Compute the classic atan2 function between y and x'''
    arg1 = y / (tf.sqrt(tf.pow(y,2) + tf.pow(x,2)) + x)
    z1 = 2*tf.atan(arg)
    
    arg2 = (tf.sqrt(tf.pow(y,2) + tf.pow(x,2)) - x) / y
    z2 = 2*tf.atan(arg)
    
    z_ = tf.select(x>0,z1,z2)
    z = tf.select(tf.equal(y,0),np.pi*tf.ones_like(z_),z_)
    return z

def get_signed_angle(u,v):
    '''Get the signed angle from one vector to another'''
    angle = atan2(u[:,1],u[:,0]) - atan2(v[:,1],v[:,0])
    return tf.mod(angle, np.pi)

def get_weights(filter_shape, collection=None, name=''):
    W_init = tf.random_normal(filter_shape)
    W = tf.Variable(W_init, collections=collection, name=name)
    return W