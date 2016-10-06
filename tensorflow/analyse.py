'''Visualize and test phase invariant features'''

import os
import sys
import time

import numpy as np

from example_scripts import *

def reload_model(saver, sess, saveDir, reloadFlag):
	"""Reload from a checkpoint"""
	if reloadFlag:
		reload_path = saveDir + "checkpoints/model.ckpt"
		saver.restore(sess, reload_path)
		print("Model restored.")

def load_data():
	mnist_train = np.load('./data/mnist_rotation_new/rotated_train.npz')
	mnist_valid = np.load('./data/mnist_rotation_new/rotated_valid.npz')
	mnist_test = np.load('./data/mnist_rotation_new/rotated_test.npz')
	mnist_trainx, mnist_trainy = mnist_train['x'], mnist_train['y']
	mnist_validx, mnist_validy = mnist_valid['x'], mnist_valid['y']
	mnist_testx, mnist_testy = mnist_test['x'], mnist_test['y']
	return mnist_trainx, mnist_validx, mnist_testx

def visualize():
	# Load data
	X, __, __ = load_data()
	X0 = X[np.random.randint(10000),:]
	X1 = np.reshape(np.flipud(np.reshape(X0, [28,28])).T, [-1])
	X = np.stack([X0,X1], axis=0)

	# Build network
	bs = 2
	x = tf.placeholder(tf.float32, [bs, 784])
	pt = tf.placeholder(tf.bool)
	model = conv_nin(x, drop_prob=0, n_filters=10, n_classes=10, bs=bs,
					 phase_train=pt)
	
	# Run examples through
	saver = tf.train.Saver()
	with tf.Session() as sess:
		reload_model(saver, sess, './', True)
		Y = sess.run(model, feed_dict={x : X, pt : False})
	Y = Y[2]
	print Y.shape
	
	# Show
	plt.ion()
	plt.show()
	for i in xrange(10):
		plt.figure(1)
		plt.cla()
		#plt.imshow(np.reshape(X[0,:], [28,28]))
		plt.imshow(Y[0,:,:,i], cmap='gray', interpolation='nearest')
		plt.figure(2)
		plt.cla()
		#plt.imshow(np.reshape(X[1,:], [28,28]))
		plt.imshow(Y[1,:,:,i], cmap='gray', interpolation='nearest')
		plt.draw()
		
		# Measure the equivariance
		MSE = np.mean((np.flipud(Y[0,:,:,i]).T - Y[1,:,:,i])**2)
		print('MSE %f' % (MSE,))
		raw_input(i)


if __name__ == '__main__':
	visualize()