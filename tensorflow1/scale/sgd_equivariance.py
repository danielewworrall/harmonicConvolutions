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


def train(inputs, outputs, train_op, opt):
	"""Training loop"""
	global_step, t_params, f_params, lr = inputs
	loss, acc, merged = outputs
	
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
				tp, fp = el.random_transform(opt['mb_size'], opt['im_size'])
				ops = [global_step, loss, acc, merged, train_op]
				feed_dict = {t_params: tp, f_params: fp, lr: current_lr}
				gs, l, top1, summary, __ = sess.run(ops, feed_dict=feed_dict)
				
				# Summary writers, printing, checkpoint saving and termination
				train_writer.add_summary(summary, gs)
				print('[{:06d} | {:06d}] Train loss: {:03f}, Train top1: {:03f}, LR: {:0.1e}' \
						.format(int(time.time()-start), gs, l, top1, current_lr))
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
			

def validate(outputs, opt):
	"""Validate the current model"""
	loss, acc = outputs
	
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
		i = 0
		try:
			while not coord.should_stop():
				i += 1
				ops = [loss, acc]
				l, top1 = sess.run(ops)
				loss_total += l
				top1_total += top1
				print('[{:06d}] Train loss: {:03f}, Valid top1: {:03f}' \
						.format(i*opt['mb_size'], l, top1))
		except tf.errors.OutOfRangeError:
			loss_total = loss_total / (i*1.)
			top1_total = top1_total / (i*1.)
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
			coord.join(threads)
	
	print('Validation loss: {:03f}, Validation accuracy: {:03f}' \
			.format(loss_total,top1_total))


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
	opt['mb_size'] = 64
	opt['n_channels'] = 32
	opt['n_iterations'] = 100000
	opt['lr_schedule'] = [50000,75000]
	opt['lr'] = 1e-3
	opt['n_labels'] = 100
	opt['save_step'] = 100
	opt['im_size'] = (256,256)
	opt['weight_decay'] = 0.0005
	opt['summary_path'] = dir_ + '/summaries/train_{:04d}'.format(opt['n_labels'])
	opt['save_path'] = dir_ + '/checkpoints/train_{:04d}/model.ckpt'.format(opt['n_labels'])
	opt['train_folder'] = opt['root'] + '/Data/ImageNet/labels/top_k/train_{:04d}'.format(opt['n_labels'])
	opt['valid_folder'] = opt['root'] + '/Data/ImageNet/labels/top_k/validation_{:04d}'.format(opt['n_labels'])
	opt['equivariant_weight'] = 0. #1e-3
	opt['is_training'] = True
	
	# Construct input graph
	if opt['is_training']:
		train_files = imagenet_loader.get_files(opt['train_folder'])
		x, labels = imagenet_loader.get_batches(train_files, True, opt)
		# Define variables
		global_step = tf.Variable(0, name='global_step', trainable=False)
		t_params = tf.placeholder(tf.float32, [opt['mb_size'],6], name='t_params')
		f_params = tf.placeholder(tf.float32, [opt['mb_size'],2,2], name='f_params')
		lr = tf.placeholder(tf.float32, [], name='lr')
		# Build the model
		logits, y, yr = models.siamese_model(x, t_params, f_params, opt)
	else:
		validation_files = imagenet_loader.get_files(opt['valid_folder'])
		x, labels = imagenet_loader.get_batches(validation_files, False, opt,
															 min_after_dequeue=1000,
															 num_epochs=1)
		# Build the model
		logits = models.single_model(x, opt)
	
	# Build loss and metrics
	softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
																				labels=labels)
	classification_loss = tf.reduce_mean(softmax)
	if opt['is_training']:
		equi_loss = 0.
		layer_equi_summaries = []
		for i, (y_, yr_) in enumerate(zip(y, yr)):
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
	argmax = tf.argmax(logits, axis=1)
	acc = tf.reduce_mean(tf.cast(tf.equal(argmax, labels), tf.float32))
	
	loss_summary = tf.summary.scalar('Loss', loss)
	class_summary = tf.summary.scalar('Classification Loss', classification_loss)
	equi_summary = tf.summary.scalar('Equivariant loss', equi_loss)
	reg_loss = tf.summary.scalar('Regularization loss', regularization_loss)
	acc_summary = tf.summary.scalar('Accuracy', acc)
	lr_summary = tf.summary.scalar('Learning rate', lr)
	merged = tf.summary.merge_all()
	
	if opt['is_training']:
		# Build optimizer
		optim = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
		train_op = optim.minimize(loss, global_step=global_step)
		
		inputs = [global_step, t_params, f_params, lr]
		outputs = [loss, acc, merged] 
		
		# Train
		return train(inputs, outputs, train_op, opt)
	else:
		outputs = [loss, acc]
		
		#Validate
		return validate(outputs, opt)


if __name__ == '__main__':
	opt = {}
	main(opt)
