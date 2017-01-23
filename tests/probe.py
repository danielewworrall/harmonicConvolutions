'''Probe the activations of a network'''

import os
import sys
import time
sys.path.append('../')

import harmonic_network_models as hnm
import numpy as np
import tensorflow as tf


def get_data():
	from io_helpers import load_dataset
	return load_dataset('../cifar10/', 'cifar_taco')


def define_settings():
	opt = {}
	opt['model'] = getattr(hnm, 'deep_cifar')
	opt['is_classification'] = True
	opt['dim'] = 32
	opt['n_classes'] = 10 
	opt['batch_size'] = 40
	opt['std_mult'] = 1.
	opt['filter_gain'] = 2
	opt['filter_size'] = 3
	opt['n_filters'] = 4*4	# Wide ResNet
	opt['resnet_block_multiplicity'] = 3
	opt['momentum'] = 0.93
	opt['n_channels'] = 3
	return opt


def running_mean(x, running_mean, counter, increment):
	R = np.sqrt(np.sum(x**2, axis=4) + 1e-4)
	mean = np.mean(x, axis=(1,2))
	mean_sum = np.sum(mean, axis=0)
	running_mean = (counter*running_mean + mean_sum)/(counter+increment)
	return running_mean


def running_power(x, running_power, counter, increment):
	R = np.sum(x**2, axis=4)
	power = np.mean(R, axis=(1,2))
	mean_power = np.sum(power, axis=0)
	running_power = (counter*running_power + mean_power)/(counter+increment)
	return running_power
	

def main():
	bs = 40
	opt = define_settings()
	is_training = tf.placeholder(tf.bool, name='is_training')
	x = tf.placeholder(tf.float32, [bs,3072], name='x')
	y = hnm.deep_cifar(opt, x, is_training, device='/gpu:0')
	
	data = get_data()
	train_x = data['train_x']
	trsh = train_x.shape
	permutation = np.random.permutation(trsh[0])
	train_x = train_x[permutation,...]
	
	rm = [0,0,0]
	rp = [0,0,0]
	counter = 0
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op, feed_dict={is_training: True})
		
		for batch in np.split(train_x, trsh[0]/bs):
			Y, activations = sess.run(y, feed_dict={x : batch, is_training : True})
			for i in xrange(3):
				rm[i] = running_mean(activations[i], rm[i], counter, bs)
				rp[i] = running_power(activations[i], rp[i], counter, bs)
			counter += bs
		
		for i in xrange(3):
			print np.sqrt(rp[i] - rm[i]**2)


if __name__ == '__main__':
	main()





































