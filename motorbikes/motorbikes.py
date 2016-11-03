'''Motorbikes testing'''

import os
import sys
import time
sys.path.append('../')

import cv2
import numpy as np
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import skimage.io as skio
import tensorflow as tf

from steer_conv import *

from matplotlib import pyplot as plt

##### MODELS #####
def deep_motorbike(x, opt):
	"""The deep_complex_bias architecture. Current test time score is 94.7% for
	7 layers deep, 5 filters
	"""
	# Sure layers weight & bias
	order = 3
	nf = opt['n_filters']
	n_classes = opt['n_classes']
	bs = opt['batch_size']
	phase_train = opt['phase_train']
	std_mult = opt['std_mult']
	filter_gain = opt['filter_gain']
	
	weights = {
		'w1' : get_weights_dict([[6,],[5,],[5,]], 1, nf, std_mult=std_mult, name='W1'),
		'w2' : get_weights_dict([[6,],[5,],[5,]], nf, nf, std_mult=std_mult, name='W2'),
		'w3' : get_weights_dict([[6,],[5,],[5,]], nf, n_classes, std_mult=std_mult, name='W3'),
	}
	
	biases = {
		'b1' : get_bias_dict(nf, order, name='b1'),
		'b2' : get_bias_dict(nf, order, name='b2')
	}
	
	psis = {
		'p1' : get_phase_dict(1, nf, order, name='psi1'),
		'p2' : get_phase_dict(nf, nf, order, name='psi2'),
		'p3' : get_phase_dict(nf, n_classes, order, name='psi3'),
	}
	
	# Convolutional Layers
	with tf.name_scope('convolutions') as scope:
		cv1 = real_input_rotated_conv(x, weights['w1'], psis['p1'],
									  filter_size=5, padding='SAME', name='1')
		cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
		
		# LAYER 2
		cv2 = complex_input_rotated_conv(cv1, weights['w2'], psis['p2'],
										 filter_size=5,
										 output_orders=xrange(order+1),
										 padding='SAME', name='2')
		cv2 = complex_batch_norm(cv2, tf.nn.relu, phase_train)

		# LAYER 3
		cv3 = complex_input_rotated_conv(cv2, weights['w3'], psis['p3'],
										 filter_size=5, padding='SAME',
										 name='3')
	
	with tf.name_scope('output') as scope:
		return tf.reduce_mean(sum_magnitudes(cv3), reduction_indices=[1,2])

################################################################################
def conv2d(X, V, b=None, strides=(1,1,1,1), padding='VALID', name='conv2d'):
    """conv2d wrapper. Supply input X, weights V and optional bias"""
    VX = tf.nn.conv2d(X, V, strides=strides, padding=padding, name=name+'_')
    if b is not None:
        VX = tf.nn.bias_add(VX, b)
    return VX

def maxpool2d(X, k=2):
    """Tied max pool. k is the stride and pool size"""
    return tf.nn.max_pool(X, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def get_weights_dict(comp_shape, in_shape, out_shape, std_mult=0.4, name='W'):
	"""Return a dict of weights for use with real_input_equi_conv. comp_shape is
	a list of the number of elements per Fourier base. For 3x3 weights use
	[3,2,2,2]. I currently assume order increasing from 0.
	"""
	weights_dict = {}
	for i, cs in enumerate(comp_shape):
		shape = cs + [in_shape,out_shape]
		weights_dict[i] = get_weights(shape, std_mult=std_mult,
									  name=name+'_'+str(i))
	return weights_dict

def get_bias_dict(n_filters, order, name='b'):
	"""Return a dict of biases"""
	bias_dict = {}
	for i in xrange(order+1):
		bias = tf.Variable(tf.constant(1e-2, shape=[n_filters]),
						   name=name+'_'+str(i))
		bias_dict[i] = bias
	return bias_dict

def get_phase_dict(n_in, n_out, order, name='b'):
	"""Return a dict of phase offsets"""
	phase_dict = {}
	for i in xrange(order+1):
		init = np.random.rand(1,1,n_in,n_out) * 2. *np.pi
		init = np.float32(init)
		phase = tf.Variable(tf.constant(init, shape=[1,1,n_in,n_out]),
						   name=name+'_'+str(i))
		phase_dict[i] = phase
	return phase_dict


##### CUSTOM FUNCTIONS FOR MAIN SCRIPT #####
def minibatcher(inputs, targets, batch_size, shuffle=False):
	"""Input and target are minibatched. Returns a generator"""
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield inputs[excerpt], targets[excerpt]

def save_model(saver, saveDir, sess, saveSubDir=''):
	"""Save a model checkpoint"""
	dir_ = saveDir + "checkpoints/" + saveSubDir
	if not os.path.exists(dir_):
		os.mkdir(dir_)
		print("Created: %s" % (dir_))
	save_path = saver.save(sess, dir_ + "/model.ckpt")
	print("Model saved in file: %s" % save_path)

def restore_model(saver, saveDir, sess):
	"""Save a model checkpoint"""
	save_path = saver.restore(sess, saveDir + "checkpoints/model.ckpt")
	print("Model restored from file: %s" % save_path)

def rotate_feature_maps(X, n_angles):
	"""Rotate feature maps"""
	X = np.reshape(X, [28,28])
	X_ = []
	for angle in np.linspace(0, 360, num=n_angles):
		X_.append(sciint.rotate(X, angle, reshape=False))
	X_ = np.stack(X_, axis=0)
	X_ = np.reshape(X_, [-1,784])
	return X_

def get_learning_rate(current, best, counter, learning_rate, delay=15):
    """If have not seen accuracy improvement in delay epochs, then divide 
    learning rate by 10
    """
    if current > best:
        best = current
        counter = 0
    elif counter > delay:
        learning_rate = learning_rate / 10.
        counter = 0
    else:
        counter += 1
    return (best, counter, learning_rate)

def dataGen(addresses, labels, batch_size, shuffle=True):
	"""Get images, resize and preprocess"""
	order = np.arange(len(addresses))
	if shuffle:
		order = np.random.permutation(order)
	batches = np.array_split(order, np.ceil(len(addresses)/(1.*batch_size)))
	for batch in batches:
		if len(batch) == batch_size:
			im = []
			lab = []
			for i in batch:
				img = np.reshape(skio.imread(addresses[i]), (125,200,1))
				img = ZMUV(img)
				im.append(img)
				lab.append(int(labels[i]))
			im = np.stack(im, axis=0)
			lab = np.hstack(lab).astype(np.int32)
			# Really need to add some kind of preprocessing
			yield (im, lab)

def ZMUV(image):
	"""Return the zero mean, unit variance image"""
	return (image - np.mean(image)) / (np.std(image) + 1e-4)

def threadedGen(generator, num_cached=50):
    """Threaded generator to multithread the data loading pipeline"""
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()

def get_addresses(base_dir, n_images, tag, pad):
	"""Return a list of addresses of form base_dir/tag_num, where tag is a
	zero-padded int of pad size pad
	"""
	addresses = []
	for i in xrange(n_images):
		addresses.append(base_dir + '/' + tag + '_' + str(i+1).zfill(pad) + '.png')
	return addresses
	

##### MAIN SCRIPT #####
def run(opt):
	tf.reset_default_graph()
	# Load dataset
	n_pos = 65
	n_neg = 900
	n_test1 = 69
	n_test2 = 100
	pos_dir = '../data/motorbikes/train_images/positives_resized'
	neg_dir = '../data/motorbikes/train_images/negatives_resized'
	test1_dir = '../data/motorbikes/test_images/test1/images_resized'
	test2_dir = '../data/motorbikes/test_images/test2/images_resized'
	
	pos_addresses = get_addresses(pos_dir, n_pos, 'image', 4)
	neg_addresses = get_addresses(neg_dir, n_neg, 'image', 4)
	test1_addresses = get_addresses(test1_dir, n_test1, 'image', 4)
	test2_addresses = get_addresses(test2_dir, n_test2, 'image', 4)
	
	oversample_ratio = n_neg/n_pos
	addresses = pos_addresses*oversample_ratio + neg_addresses
	labels = np.asarray([1,]*n_pos*oversample_ratio + [0,]*n_neg)

	# Parameters
	nesterov=True
	model = opt['model']
	lr = opt['lr']
	batch_size = opt['batch_size']
	n_epochs = opt['n_epochs']
	n_filters = opt['n_filters']
	trial_num = opt['trial_num']
	combine_train_val = opt['combine_train_val']
	std_mult = opt['std_mult']
	filter_gain = opt['filter_gain']
	momentum = opt['momentum']
	psi_preconditioner = opt['psi_preconditioner']
	delay = opt['delay']
	summary_dir = opt['log_dir'] + '/trial' + str(trial_num)
	checkpoint_dir = opt['checkpoint_dir'] + '/trial' + str(trial_num)
	
	# Network Parameters
	n_classes = opt['n_classes']
	dataset_size = len(labels)
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [batch_size, 125, 200, 1])
	y = tf.placeholder(tf.int64, [batch_size])
	learning_rate = tf.placeholder(tf.float32)
	#keep_prob = tf.placeholder(tf.float32)
	phase_train = tf.placeholder(tf.bool)
	opt['phase_train'] = phase_train
	
	# Construct model
	pred = deep_motorbike(x, opt)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
	print('  Constructed loss')
	opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,
									 momentum=momentum,
									 use_nesterov=nesterov).minimize(cost)
	print('  Optimizer built')
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	print('  Evaluation metric constructed')
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	print('  Variables initialized')
	
	# Summary writers
	acc_ph = tf.placeholder(tf.float32, [], name='acc_')
	acc_op = tf.scalar_summary("Validation Accuracy", acc_ph)
	cost_ph = tf.placeholder(tf.float32, [], name='cost_')
	cost_op = tf.scalar_summary("Training Cost", cost_ph)
	lr_ph = tf.placeholder(tf.float32, [], name='lr_')
	lr_op = tf.scalar_summary("Learning Rate", lr_ph)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = False
	config.inter_op_parallelism_threads = 1 #prevent inter-session threads?
	sess = tf.Session(config=config)
	
	if not os.path.exists(checkpoint_dir):
		os.mkdir(checkpoint_dir)
		print("Created: %s" % (summary_dir))
	if not os.path.exists(summary_dir):
		os.mkdir(summary_dir)
		print("Created: %s" % (summary_dir))
	summary = tf.train.SummaryWriter(summary_dir, sess.graph)
	print('  Summaries constructed')
	
	# Launch the graph
	sess.run(init)
	saver = tf.train.Saver()
	start = time.time()
	lr_current = lr
	step = 0.
	counter = 0
	best = 0.
	print('  Begin training')
	# Keep training until reach max iterations
	epoch = 0
	while epoch < n_epochs:
		generator = dataGen(addresses, labels, batch_size, shuffle=True)
		cost_total = 0.
		acc_total = 0.
		for i, batch in enumerate(generator):
			batch_x, batch_y = batch
			# Optimize
			feed_dict = {x: batch_x, y: batch_y, learning_rate: lr_current,
						 phase_train : True}
			__, cost_, acc_ = sess.run([opt, cost, accuracy],
				feed_dict=feed_dict)
			if np.isnan(cost_):
				print
				print('Oops: Training went unstable')
				print
				return -1
				
			cost_total += cost_
			acc_total += acc_
			step += 1
		cost_total /=(i+1.)
		acc_total /=(i+1.)
		
		feed_dict={cost_ph : cost_total, acc_ph : acc_total, lr_ph : lr_current}
		summaries = sess.run([cost_op, acc_op, lr_op], feed_dict=feed_dict)
		for summ in summaries:
			summary.add_summary(summ, step)

		best, counter, lr_current = get_learning_rate(acc_total, best, counter, lr_current, delay=delay)
		
		print "[" + str(trial_num),str(epoch) + \
			"], Minibatch Loss: " + \
			"{:.6f}".format(cost_total) + ", Train Acc: " + \
			"{:.5f}".format(acc_total) + ", Time: " + \
			"{:.5f}".format(time.time()-start) + ", Counter: " + \
			"{:2d}".format(counter) 
		epoch += 1
		
		if epoch % 10 == 0:
			save_path = saver.save(sess, checkpoint_dir + '/model.ckpt')
			print("Model saved in file: %s" % save_path)
	
	save_path = saver.save(sess, checkpoint_dir + '/model.ckpt')
	print("Model saved in file: %s" % save_path)
	sess.close()


if __name__ == '__main__':
	opt = {}
	opt['model'] = 'deep_motorbike'
	opt['n_classes'] = 2
	opt['lr'] = 3e-2
	opt['batch_size'] = 30
	opt['n_epochs'] = 120
	opt['n_filters'] = 8
	opt['trial_num'] = 'M'
	opt['combine_train_val'] = False
	opt['std_mult'] = 0.3
	opt['filter_gain'] = 3.7
	opt['momentum'] = 0.93
	opt['psi_preconditioner'] = 3.4
	opt['delay'] = 13
	opt['log_dir'] = '../logs/motorbikes'
	opt['checkpoint_dir'] = '../checkpoints/motorbikes'
	run(opt)

