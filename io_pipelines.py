import os
import sys
import zipfile
import urllib2

import numpy as np

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
			})

		#decode (will still be a string at this point)
		x = tf.decode_raw(features['x_raw'], tf.float32, name="decodeX")

		#decode (will still be a string at this point)
		y = tf.decode_raw(features['y_raw'], np.float32, name="decodeY")

	return x, y

def pipeline(fileNames, batch_size, num_epochs, data_aug_function, shuffle=True):
	with tf.name_scope('IO') as scope:
		#create a string queue in case the dataset is split or we are combining
		filename_string_queue = tf.train.string_input_producer(fileNames)

		#add nodes to read and decode a single example
		image, label = read_decode(filename_string_queue)

		#add user-specified nodes for augmentation
		image, label = data_aug_function(image, label)

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