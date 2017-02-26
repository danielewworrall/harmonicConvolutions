'''ImageNet loader'''

import os
import sys
import time

import tensorflow as tf


def read_my_file_format(filename_queue):
	# Text file reader
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [[""],[""]]
	address, label = tf.decode_csv(value, record_defaults=record_defaults)
	address = '/home/dworrall/Data/ImageNet/'+address
	# Image reader
	file_contents = tf.read_file(address)
	image = tf.image.decode_jpeg(file_contents, channels=3)
	# TO DO
	processed_image = tf.image.resize_image_with_crop_or_pad(image,227,227)
	return processed_image, tf.string_to_number(label)


def main(files, read_threads):	
	filename_queue = tf.train.string_input_producer(files, shuffle=True)
	image, label = read_my_file_format(filename_queue)
	image_batch, label_batch = tf.train.shuffle_batch(
		[image, label], batch_size=32, num_threads=4, shapes=[[227,227,3],[]],
		capacity=5000, min_after_dequeue=1000)
	
	with tf.Session() as sess:
		# Start populating the filename queue.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
	
		for i in xrange(1200):
			# Retrieve a single instance:
			a, l = sess.run([image_batch, label_batch])
			print i*32
		
			coord.request_stop()
			coord.join(threads)


if __name__ == '__main__':
	files = ["/home/dworrall/Data/ImageNet/labels/subsets/train_0032.txt"]
	main(files, 4)