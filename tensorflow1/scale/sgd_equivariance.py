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


def train(inputs, outputs, ops, opt):
	"""Training loop"""
	global_step, t_params, f_params, lr, is_training = inputs
	loss, top1, top5, merged = outputs
	train_op = ops
	
	# For checkpoints
	saver = tf.train.Saver()
	gs = 0
	start = time.time()
	
	with tf.Session() as sess:
		# Threading and queueing
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		# Initialize variables
		init = tf.global_variables_initializer()
		sess.run(init)
		
		train_writer = tf.summary.FileWriter(opt['summary_path'], sess.graph)
		# Training loop
		try:
			while not coord.should_stop():
				# Learning rate
				exponent = sum([gs > i for i in opt['lr_schedule']])
				current_lr = opt['lr']*np.power(0.1, exponent)
				
				# Run training steps
				#tp, fp = el.random_transform(opt['mb_size'], opt['im_size'])
				tp, fp = el.random_transform_6(opt['mb_size'], opt['im_size'])
				ops = [global_step, loss, top1, top5, merged, train_op]
				feed_dict = {t_params: tp, f_params: fp, lr: current_lr, is_training: True}
				gs, l, t1, t5, summary, __ = sess.run(ops, feed_dict=feed_dict)
				
				# Summary writers, printing, checkpoint saving and termination
				train_writer.add_summary(summary, gs)
				print('[{:06d} | {:06d}] Train loss: {:03f}, Train top1: {:03f}, Train top5: {:03f}, LR: {:0.1e}' \
						.format(int(time.time()-start), gs, l, t1, t5, current_lr))
				if gs % opt['save_step'] == 0:
					save_path = saver.save(sess, opt['save_path'], global_step=gs)
					print("Model saved in file: %s" % save_path)
				# Escape above a certain number of iterations
				if gs > opt['n_iterations']:
					break
		except tf.errors.OutOfRangeError:
			pass
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
			coord.join(threads)
		
		sess.close()
			

def validate(inputs, outputs, opt):
	"""Validate the current model"""
	is_training = inputs
	loss, top1, top5 = outputs
	
	# For checkpoints
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		init_op = tf.local_variables_initializer()
		sess.run(init_op)
		
		# Threading and queueing
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		# Restore model
		model_file = get_latest_model(opt['save_path'])
		saver.restore(sess, model_file)
		print('Model restored: {:s}'.format(model_file))
		
		# Validation loop
		loss_total = 0.
		top1_total = 0.
		top5_total = 0.
		i = 0
		try:
			while not coord.should_stop():
				i += 1
				feed_dict = {is_training: False}
				ops = [loss, top1, top5]
				l, t1, t5 = sess.run(ops, feed_dict=feed_dict)
				loss_total += l
				top1_total += t1
				top5_total += t5
				print('[{:06d}] Train loss: {:03f}, Valid top1: {:03f}, Valid top5: {:03f}' \
						.format(i*opt['mb_size'], l, t1, t5))
		except tf.errors.OutOfRangeError:
			loss_total = loss_total / (i*1.)
			top1_total = top1_total / (i*1.)
			top5_total = top5_total / (i*1.)
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
			coord.join(threads)
	
	print('Validation loss: {:03f}, Validation top1: {:03f}, Validation top5: {:03f}' \
			.format(loss_total,top1_total,top5_total))


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
	opt['n_channels'] = 60
	opt['n_iterations'] = 50000
	opt['lr_schedule'] = [35000,45000]
	opt['lr'] = 1e-2
	opt['n_labels'] = 1000
	opt['density'] = 32
	opt['save_step'] = 100
	opt['im_size'] = (224,224)
	opt['weight_decay'] = 1e-5
	opt['equivariant_weight'] = 1e-3
	opt['equivariance_end'] = 3
	flag = 'bn'
	opt['summary_path'] = dir_ + '/summaries/subsets/train_{:04d}_{:.0e}_{:s}'.format(opt['density'], opt['equivariant_weight'], flag)
	opt['save_path'] = dir_ + '/checkpoints/subsets/train_{:04d}_{:.0e}_{:s}/model.ckpt'.format(opt['density'], opt['equivariant_weight'], flag)
	opt['train_folder'] = opt['root'] + '/Data/ImageNet/labels/subsets/train_{:04d}'.format(opt['density'])
	opt['valid_folder'] = opt['root'] + '/Data/ImageNet/labels/subsets/validation_{:04d}'.format(opt['density'])
	opt['is_training'] = True
	
	
	if not os.path.isdir(os.path.dirname(opt['save_path'])):
		os.mkdir(os.path.dirname(opt['save_path']))
		print('Created directory: {:s}'.format(os.path.dirname(opt['save_path'])))
	
	# Construct input graph
	if opt['is_training']:
		train_files = imagenet_loader.get_files(opt['train_folder'])
		x, labels = imagenet_loader.get_batches(train_files, True, opt)
		# Define variables
		global_step = tf.Variable(0, name='global_step', trainable=False)
		t_params = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
		f_params = tf.placeholder(tf.float32, [opt['mb_size'],6,6], name='f_params')
		lr = tf.placeholder(tf.float32, [], name='lr')
		is_training = tf.placeholder(tf.bool, [], name='is_training')
		# Build the model
		logits, y, yr = models.siamese_model(x, is_training, t_params, f_params, opt)
	else:
		is_training = tf.placeholder(tf.bool, [], name='is_training')
		validation_files = imagenet_loader.get_files(opt['valid_folder'])
		x, labels = imagenet_loader.get_batches(validation_files, False, opt,
															 min_after_dequeue=1000,
															 num_epochs=1)
		# Build the model
		logits = models.single_model(x, is_training, opt)
	
	# Build loss and metrics
	softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
																				labels=labels)
	classification_loss = tf.reduce_mean(softmax)
	if opt['is_training']:
		equi_loss = 0.
		layer_equi_summaries = []
		for i, (y_, yr_) in enumerate(zip(y, yr)):
			if i < opt['equivariance_end']:
				print('Adding equivariance regularizer to layer: {:d}'.format(i+1))
				layer_equi_loss = tf.reduce_mean(tf.square(y_ - yr_))
				equi_loss += layer_equi_loss
				layer_equi_summaries.append(tf.summary.scalar('Equivariant loss'+str(i), layer_equi_loss))
		loss = classification_loss + opt['equivariant_weight']*equi_loss
	else:
		loss = classification_loss
	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	regularization_loss = tf.add_n(regularization_losses)
	loss += regularization_loss
		
	# Accuracies
	top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
	top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
	
	if opt['is_training']:
		loss_summary = tf.summary.scalar('Loss', loss)
		class_summary = tf.summary.scalar('Classification Loss', classification_loss)
		equi_summary = tf.summary.scalar('Equivariant loss', equi_loss)
		reg_loss = tf.summary.scalar('Regularization loss', regularization_loss)
		top1_summary = tf.summary.scalar('Top1 Accuracy', top1)
		top5_summary = tf.summary.scalar('Top 5 Accuracy', top5)
		lr_summary = tf.summary.scalar('Learning rate', lr)
		merged = tf.summary.merge_all()
	
	if opt['is_training']:
		# Build optimizer
		optim = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
		train_op = optim.minimize(loss, global_step=global_step)
		
		inputs = [global_step, t_params, f_params, lr, is_training]
		outputs = [loss, top1, top5, merged]
		ops = [train_op]
		
		# Train
		return train(inputs, outputs, ops, opt)
	else:
		inputs = is_training
		outputs = [loss, top1, top5]
		
		#Validate
		return validate(inputs, outputs, opt)


if __name__ == '__main__':
	opt = {}
	main(opt)
