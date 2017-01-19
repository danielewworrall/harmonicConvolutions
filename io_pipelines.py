import os
import sys
import zipfile
import urllib2

import numpy as np

import tensorflow as tf

def read_decode(opt, data, filename_string_queue):
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
		#decode x and y (will still be strings at this point)
		x, y = data['data_decode_function'](features)
		return x, y

def pipeline(fileNames, opt, data, shuffle=True):
	with tf.name_scope('IO') as scope:
		#create a string queue in case the dataset is split or we are combining
		filename_string_queue = tf.train.string_input_producer(fileNames)

		#add nodes to read and decode a single example
		image, label = read_decode(opt, data, filename_string_queue)

		#add user-specified nodes for augmentation
		image, label = data['data_process_function'](image, label)

		#reshape, treating the input as an image and label as generic tensor
		#we need to do this because shuffling with queues requires static tensor
		#size knowledge
		if len(data['x_target_shape']) > 0:
			image = tf.reshape(image, data['x_target_shape'])

		if len(data['y_target_shape']) > 0:
			label = tf.reshape(label, data['y_target_shape'])
			label = tf.squeeze(label) #remove singleton dimensions

		min_after_capacity = 10000
		capacity = min_after_capacity * opt['batch_size'] * 4
		if shuffle:
			image_batch, label_batch = tf.train.shuffle_batch(
				[image, label], batch_size=opt['batch_size'],
				capacity=capacity, min_after_dequeue=min_after_capacity)
		else:
			image_batch, label_batch = tf.train.batch([image,
				label], batch_size=opt['batch_size'], capacity=capacity)

	return image_batch, label_batch