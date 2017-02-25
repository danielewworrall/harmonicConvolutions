import os
import sys
import zipfile
import urllib2

import numpy as np

import tensorflow as tf

def read_decode(filename_string_queue):
	 with tf.name_scope('IO') as scope:
		reader = tf.TFRecordReader()
		#get next example
		key, example = reader.read(filename_string_queue)
		#parse
		features = tf.parse_single_example(serialized=example,
			features={
			'x_raw': tf.FixedLenFeature([], tf.string),
			'y_raw': tf.FixedLenFeature([], tf.string),
			#x size
			'x_shape': tf.FixedLenFeature([], tf.string),
			#y size
			'y_shape': tf.FixedLenFeature([], tf.string),
			})
		#decode xs (will still be a string at this point)
		x = tf.decode_raw(features['x_raw'], tf.float32, name="decodeX")
		x_shape = tf.reshape(tf.decode_raw(features['x_shape'], tf.int64, name='decode_x_shape'), [3])
		x = tf.reshape(x, tf.cast(x_shape, tf.int32))

		#decode ys (will still be a string at this point)
		y = tf.decode_raw(features['y_raw'], np.float32, name="decodeY")
		y_shape = tf.reshape(tf.decode_raw(features['y_shape'], tf.int64, name='decode_y_shape'), [3])
		y = tf.reshape(y, tf.cast(y_shape, tf.int32))
		y = tf.cast(y, tf.int64)
		return x, y

def pipeline(fileNames, batch_size, num_epochs, data_aug_function,
	x_target_shape, y_target_shape, shuffle=True):
	with tf.name_scope('IO') as scope:
		#create a string queue in case the dataset is split or we are combining
		filename_string_queue = tf.train.string_input_producer(fileNames)

		#add nodes to read and decode a single example
		image, label = read_decode(filename_string_queue)

		#add user-specified nodes for augmentation
		image, label = data_aug_function(image, label)

		#reshape, treating the input as an image and label as generic tensor
		#we need to do this because shuffling with queues requires static tensor
		#size knowledge
		if len(x_target_shape) > 0:
			#image = tf.image.resize_image_with_crop_or_pad(image, x_target_shape[0], x_target_shape[1])
			image = tf.reshape(image, x_target_shape + [1, 1])

		if len(y_target_shape) > 0:
			label = tf.reshape(label, y_target_shape)
			label = tf.squeeze(label) #remove singleton dimensions

		min_after_capacity = 10000
		capacity = min_after_capacity * batch_size * 4
		if shuffle:
			image_batch, label_batch = tf.train.shuffle_batch(
				[image, label], batch_size=batch_size,
				capacity=capacity, min_after_dequeue=min_after_capacity)
		else:
			image_batch, label_batch = tf.train.batch([image,
				label], batch_size=batch_size, capacity=capacity)

	return image_batch, label_batch