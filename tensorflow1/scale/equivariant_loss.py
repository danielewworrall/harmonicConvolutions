'''Transformer loss'''

import os
import sys
import time

import numpy as np

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
	x = feature_space_transform4d(x,xsh)
	# 2) Rotate features spatially
	y = transformer(x, t_params, (xsh[1],xsh[2]))
	return y


def get_t_transform(theta, imsh):
	scale1 = np.array([[float(imsh[0])/imsh[1], 0.], [0., 1.]])
	scale2 = np.array([[float(imsh[1])/imsh[0], 0.], [0., 1.]])
	rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	mat = np.dot(scale1, rot)
	mat = np.dot(mat, scale2)
	
	mat = np.hstack((mat,np.zeros((2,1))))
	mat = mat.astype('float32')
	mat = mat.flatten()
	return mat


def get_f_transform(theta):
	rot = np.array([[np.cos(theta), -np.sin(theta)],
						[np.sin(theta), np.cos(theta)]])
	return rot


def random_transform(mb_size, imsh):
	t_params = []
	f_params = []
	for t in np.random.rand(mb_size):
		t_params.append(get_t_transform(2*np.pi*t, (imsh[0],imsh[1])))
		f_params.append(get_f_transform(2*np.pi*t))
	return np.vstack(t_params), np.stack(f_params, axis=0)