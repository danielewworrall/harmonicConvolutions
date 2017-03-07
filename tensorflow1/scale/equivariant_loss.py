'''Transformer loss'''

import os
import sys
import time

import numpy as np
import scipy.linalg as splin

import tensorflow as tf

from spatial_transformer import transformer


def feature_space_transform2d(x, xsh, f_params):
	x = tf.reshape(x, tf.stack([xsh[0],xsh[1]/2,2]))
	f1 = tf.reshape(f_params[:,0,:], tf.stack([xsh[0],1,2]))
	f2 = tf.reshape(f_params[:,1,:], tf.stack([xsh[0],1,2]))
	x0 = tf.reduce_sum(tf.multiply(x, f1), axis=2)
	x1 = tf.reduce_sum(tf.multiply(x, f2), axis=2)
	x = tf.stack([x0, x1], axis=-1)
	return tf.reshape(x, tf.stack([xsh[0],xsh[1]]))


def feature_space_transform4d(x, xsh, f_params):
	"""Perform a block matrix multiplication on the last dimension of a tensor"""
	x = tf.reshape(x, tf.stack([xsh[0],xsh[1],xsh[2],xsh[3]/2,2]))
	f1 = tf.reshape(f_params[:,0,:], tf.stack([xsh[0],1,1,1,2]))
	f2 = tf.reshape(f_params[:,1,:], tf.stack([xsh[0],1,1,1,2]))
	x0 = tf.reduce_sum(tf.multiply(x, f1), axis=4)
	x1 = tf.reduce_sum(tf.multiply(x, f2), axis=4)
	x = tf.stack([x0, x1], axis=-1)
	return tf.reshape(x, tf.stack([xsh[0],xsh[1],xsh[2],xsh[3]]))


def transform_features(x, t_params, f_params):
	"""Rotate features in the channels"""
	# 1) Rotate features through channels. We have to perform a broadcasted
	# matrix--matrix multiply on two subarrays of the whole tensor, but this does
	# not currently exist in TensorFlow, so we have to do it the long way.
	xsh = x.get_shape().as_list()
	x = feature_space_transform4d(x,xsh,f_params)
	# 2) Rotate features spatially
	y = transformer(x, t_params, (xsh[1],xsh[2]))
	return y


def get_t_transform(theta, imsh, scale_x=1., scale_y=1.):
	scale1 = np.array([[float(imsh[0])/imsh[1], 0.], [0., 1.]])
	scale2 = np.array([[float(imsh[1])/imsh[0], 0.], [0., 1.]])
	rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	Scale = np.array([[scale_x, 0.], [0., scale_y]])
	linear = np.dot(Scale, rot)
	
	mat = np.dot(scale1, linear)
	mat = np.dot(mat, scale2)
	
	mat = np.hstack((mat,np.zeros((2,1))))
	mat = mat.astype('float32')
	mat = mat.flatten()
	return mat


def get_f_transform(theta):
	Rot = np.array([[np.cos(theta), -np.sin(theta)],
						[np.sin(theta), np.cos(theta)]])	
	return Rot


def random_transform(mb_size, imsh):
	t_params = []
	f_params = []
	for __ in xrange(mb_size):
		t = np.random.rand()
		t_params.append(get_t_transform(2*np.pi*t, (imsh[0],imsh[1])))
		f_params.append(get_f_transform(2*np.pi*t))
	return np.vstack(t_params), np.stack(f_params, axis=0)



def transform_features_6(x, t_params, f_params):
	"""Rotate features in the channels"""
	# 1) Rotate features through channels. We have to perform a broadcasted
	# matrix--matrix multiply on two subarrays of the whole tensor, but this does
	# not currently exist in TensorFlow, so we have to do it the long way.
	xsh = x.get_shape().as_list()
	x = feature_space_transform4d_6(x, xsh, f_params)
	# 2) Rotate features spatially
	y = transformer(x, t_params, (xsh[1],xsh[2]))
	return y


def feature_space_transform4d_6(x, xsh, f_params):
	"""Perform a block matrix multiplication on the last dimension of a tensor,
	operating on a 6D representation
	"""
	x = tf.reshape(x, tf.stack([xsh[0],xsh[1],xsh[2],xsh[3]/6,6]))
	f1 = tf.reshape(f_params[:,0,:], tf.stack([xsh[0],1,1,1,6]))
	f2 = tf.reshape(f_params[:,1,:], tf.stack([xsh[0],1,1,1,6]))
	f3 = tf.reshape(f_params[:,2,:], tf.stack([xsh[0],1,1,1,6]))
	f4 = tf.reshape(f_params[:,3,:], tf.stack([xsh[0],1,1,1,6]))
	f5 = tf.reshape(f_params[:,4,:], tf.stack([xsh[0],1,1,1,6]))
	f6 = tf.reshape(f_params[:,5,:], tf.stack([xsh[0],1,1,1,6]))
	x1 = tf.reduce_sum(tf.multiply(x, f1), axis=4)
	x2 = tf.reduce_sum(tf.multiply(x, f2), axis=4)
	x3 = tf.reduce_sum(tf.multiply(x, f3), axis=4)
	x4 = tf.reduce_sum(tf.multiply(x, f4), axis=4)
	x5 = tf.reduce_sum(tf.multiply(x, f5), axis=4)
	x6 = tf.reduce_sum(tf.multiply(x, f6), axis=4)
	x = tf.stack([x1, x2, x3, x4, x5, x6], axis=-1)
	return tf.reshape(x, tf.stack([xsh[0],xsh[1],xsh[2],xsh[3]]))



def get_f_transform_6(theta, scale_x=1., scale_y=1.):
	Rot = np.array([[np.cos(theta), -np.sin(theta)],
						[np.sin(theta), np.cos(theta)]])
	Scale_x = np.array([[np.cos(scale_x), -np.sin(scale_x)],
								[np.sin(scale_x), np.cos(scale_x)]])
	Scale_y = np.array([[np.cos(scale_y), -np.sin(scale_y)],
								[np.sin(scale_y), np.cos(scale_y)]])
	return splin.block_diag(Rot, Scale_x, Scale_y)


def random_transform_6(mb_size, imsh):
	t_params = []
	f_params = []
	
	def sc(x, u, l):
		return (u-l)*x + l
	
	for __ in xrange(mb_size):
		t = np.random.rand()
		sx = np.random.rand()
		sy = np.random.rand()
		t_params.append(get_t_transform(2*np.pi*t, (imsh[0],imsh[1]),
												  scale_x=sc(sx, 1.3, 1./1.3),
												  scale_y=sc(sy, 1.3, 1./1.3)))
		f_params.append(get_f_transform_6(2*np.pi*t,
													 scale_x=np.pi*sx,
													 scale_y=np.pi*sy))
	return np.vstack(t_params), np.stack(f_params, axis=0)












































