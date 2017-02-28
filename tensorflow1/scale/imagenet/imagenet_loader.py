'''ImageNet loader'''

import os
import sys
import time

import numpy as np
import tensorflow as tf


def get_files(folder):
	fnames = []
	for root, dirs, files in os.walk(folder):
		for f in files:
			if 'chunk' in f:
				fname = root + '/' + f
				fnames.append(fname)
	return fnames


def read_my_file_format(filename_queue, im_size, opt):
	# Text file reader
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [[""],[""]]
	address, label = tf.decode_csv(value, record_defaults=record_defaults)
	address = '/home/dworrall/Data/ImageNet/'+address

	# Image reader
	file_contents = tf.read_file(address)
	image = tf.image.decode_jpeg(file_contents, channels=3)
	image = tf.to_float(image)
	# Image preprocessing
	image = tf.image.resize_image_with_crop_or_pad(image,im_size[0],im_size[1])/255.
	if opt['is_training']:
		image = tf.image.random_flip_left_right(image)
		image = tf.image.per_image_standardization(image)
	
	return image, tf.to_int64(tf.string_to_number(label))


def get_batches(files, shuffle, opt, min_after_dequeue=1000, num_epochs=None):
	batch_size = opt['mb_size']
	im_size = opt['im_size']
	
	filename_queue = tf.train.string_input_producer(files, shuffle=shuffle,
																	num_epochs=num_epochs)
	image, label = read_my_file_format(filename_queue, im_size, opt)
	
	num_threads = 4
	#min_after_dequeue = 1000
	capacity = min_after_dequeue + (num_threads+1)*batch_size
	
	#image_batch, label_batch = tf.train.shuffle_batch(
	#	[image, label], batch_size=batch_size, num_threads=num_threads, 
	#	capacity=capacity, min_after_dequeue=min_after_dequeue)
	image_batch, label_batch = tf.train.shuffle_batch_join(
		[[image, label]], batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)
	
	return image_batch, label_batch


if __name__ == '__main__':
	train_folder = "/home/dworrall/Data/ImageNet/labels/subsets/train_0004"
	files = get_train_files(train_folder)
	main(files, 4)