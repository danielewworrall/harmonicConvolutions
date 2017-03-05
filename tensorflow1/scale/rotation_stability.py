'''SGD equivariance'''

import os
import sys
import time
sys.path.append('../')
sys.path.append('./imagenet')

import cv2
import input_data
import numpy as np
import skimage.io as skio
import tensorflow as tf

import equivariant_loss as el
import imagenet_loader
import models

from matplotlib import pyplot as plt


def get_feature_maps(inputs, outputs, opt):
	"""Validate the current model"""
	x, is_training, t_params = inputs
	features = outputs
	
	# For checkpoints
	saver = tf.train.Saver()
	feature_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

	with tf.Session() as sess:
		init_op = tf.local_variables_initializer()
		sess.run(init_op)
		
		# Restore model
		model_file = get_latest_model(opt['save_path'])
		saver.restore(sess, model_file)
		print('Model restored: {:s}'.format(model_file))

		for i in xrange(opt['n_angles']):
			angle = (2.*np.pi*i)/opt['n_angles']
			tp = el.get_t_transform(angle, opt['im_size'])	
			feed_dict = {x: X,is_training: False, t_params: tp}
			Features = sess.run(features, feed_dict=feed_dict)
			np.savez('{:s}/features_{:03f}'.format(opt['save_features_path'], angle), Features, feature_names)


def get_latest_model(model_file):
	"""Model file"""
	dirname = os.path.dirname(model_file)
	basename = os.path.basename(model_file)
	nums = []
	for root, dirs, files in os.walk(dirname):
		for f in files:
			f = f.split('-')
			if f[0] == basename:
				nums.append(int(f[1].split('.')[0]))
	model_file += '-{:05d}'.format(max(nums))
	return model_file

	
def main(opt):
	"""Main loop"""
	
	tf.reset_default_graph()
	opt['root'] = '/home/dworrall'
	dir_ = opt['root'] + '/Code/harmonicConvolutions/tensorflow1/scale'
	opt['mb_size'] = 25
	opt['n_channels'] = 64
	opt['n_iterations'] = 50000
	opt['lr_schedule'] = [25000,37500]
	opt['lr'] = 1e-2
	opt['n_labels'] = 50
	opt['save_step'] = 100
	opt['im_size'] = (224,224)
	opt['weight_decay'] = 1e-5
	opt['equivariant_weight'] = 1e-1
	opt['equivariance_end'] = 3
	opt['n_angles'] = 20
	flag = 'bn'
	opt['summary_path'] = dir_ + '/summaries/train_{:04d}_{:.0e}_{:s}'.format(opt['n_labels'], opt['equivariant_weight'], flag)
	opt['save_path'] = dir_ + '/checkpoints/train_{:04d}_{:.0e}_{:s}/model.ckpt'.format(opt['n_labels'], opt['equivariant_weight'], flag)
	opt['train_folder'] = opt['root'] + '/Data/ImageNet/labels/top_k/train_{:04d}'.format(opt['n_labels'])
	opt['valid_folder'] = opt['root'] + '/Data/ImageNet/labels/top_k/validation_{:04d}'.format(opt['n_labels'])
	opt['is_training'] = False
	opt['save_features_path'] = './feature_maps'
	
	# Construct input graph
	x = tf.placeholder(tf.float32, [1,224,224,3], name='x')
	is_training = tf.placeholder(tf.bool, [], name='is_training')
	t_params = tf.placeholder(tf.float32, [], name='t_params')

	# Build the model
	features = models.single_model_feature_maps(x, is_training, opt)
	
	# Construct io
	inputs = [x, is_training, t_params]
	outputs = features
		
	#Validate
	return get_feature_maps(inputs, outputs, opt)


if __name__ == '__main__':
	opt = {}
	main(opt)
