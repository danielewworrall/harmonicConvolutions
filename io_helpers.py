from __future__ import division
import os
from os import path
import sys
import zipfile
import urllib2
import pickle
import ntpath
import copy

import numpy as np
import scipy.ndimage.interpolation as sciint
import skimage.color as skco
import skimage.exposure as skiex
import skimage.io as skio
import skimage.transform as sktr

import tensorflow as tf

##### HELPERS #####
def checkFolder(dir):
	"""Checks if a folder exists and creates it if not.
	dir: directory
	Returns nothing
	"""
	if not os.path.exists(dir):
		os.makedirs(dir)

#the simple solution from here:
#http://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file-in-python
def save_dict(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def get_num_items_in_tfrecords_list(files, verbose=True):
	#guard against empty lists
	if len(files) == 0:
		return 0
	i = 0
	overwrite_meta = False
	meta_name = os.path.dirname(files[0]) + '/meta.plk'
	if verbose:
		print('Looking for meta-file ' + meta_name)
	#if the meta-file exists we use its information
	#we also potentially amend it for missing entries
	if os.path.isfile(meta_name):
		if verbose:
			print('Meta information found for dataset.')
		dataset_records = load_dict(meta_name)
		for record_file in files:
			if ntpath.basename(record_file) in dataset_records:
				i += dataset_records[ntpath.basename(record_file)]
			else: #count manually if this has not been cached
				overwrite_meta = True
				if verbose:
					print('WARNING: meta-file [' + meta_name \
						+ "] does not contain a record for [ " + record_file + " ], creating...")
				local_num_records = 0
				for record in tf.python_io.tf_record_iterator(record_file):
					local_num_records += 1
				dataset_records[ntpath.basename(record_file)] = local_num_records
				i += local_num_records
		if overwrite_meta:
			save_dict(dataset_records, meta_name)
		return i
	#otherwise, we can need to create it
	else:
		if verbose:
			print('No meta information for dataset found, caching for future use.')
		dataset_records = {}
		for record_file in files:
			local_num_records = 0
			if verbose:
				print("Processing [ " + record_file + " ]")
			for record in tf.python_io.tf_record_iterator(record_file):
				local_num_records += 1
			dataset_records[ntpath.basename(record_file)] = local_num_records
			i += local_num_records
		#finally save the counts to a file
		save_dict(dataset_records, meta_name)
	return i

def get_all_tfrecords(directory):
	train_files = []
	valid_files = []
	test_files = []
	for file in os.listdir(directory):
		if file.endswith(".tfrecords"):
			if file.startswith("train"):
				train_files.append(directory + '/' + file)
			elif file.startswith("valid"):
				valid_files.append(directory + '/' + file)
			elif file.startswith("test"):
				test_files.append(directory + '/' + file)
	train_files.sort()
	valid_files.sort()
	test_files.sort()
	return train_files, valid_files, test_files

def discover_and_setup_tfrecords(directory, data, use_train_fraction=1.0, use_random_subset=False):
	train_files, valid_files, test_files = get_all_tfrecords(directory)
	if len(test_files) == 0:
		test_files = copy.deepcopy(valid_files)

	num_all_train_files = get_num_items_in_tfrecords_list(train_files)

	if use_train_fraction < 1.0:
		num_examples = get_num_items_in_tfrecords_list(train_files, verbose=False)
		single_file = get_num_items_in_tfrecords_list([train_files[0]], verbose=False)
		num_examples_target = num_examples * use_train_fraction
		low = int(num_examples_target / single_file)
		high = low + 1
		if num_examples_target - (low * single_file) < num_examples_target - (high * single_file):
			num_files = low
		else:
			num_files = high
		
		perm = np.random.permutation(len(train_files))
		new_train_files = []
		for i in xrange(num_files):
			if use_random_subset: #take a random subset of files if specified
				new_train_files.append(train_files[perm[i]])
			else:
				new_train_files.append(train_files[i])
		#overwrite original array
		train_files = new_train_files
		print('Given a fraction of ' + str(use_train_fraction) + \
			', we use ' + str(num_files) + \
			' files with ' + str(get_num_items_in_tfrecords_list(train_files, verbose=False)) + \
			' number of training examples out of ' + str(num_examples))
		print('\tGiven fraction: ' + str(use_train_fraction))
		print('\tUsed fraction: ' + str(get_num_items_in_tfrecords_list(train_files, verbose=False) / num_all_train_files))
		print('\tIf this is unsatisfactory, make your tfrecords files more fine-grained.')

	data['train_files'] = train_files
	data['valid_files'] = valid_files
	data['test_files'] = test_files
	#get the number of items of each set
	data['train_items'] = get_num_items_in_tfrecords_list(data['train_files'], verbose=False)
	data['valid_items'] = get_num_items_in_tfrecords_list(data['valid_files'], verbose=False)
	data['test_items'] = get_num_items_in_tfrecords_list(data['test_files'], verbose=False)
	print('Num train examples: ', data['train_items'])
	print('Num validation examples: ', data['valid_items'])
	print('Num test examples: ', data['test_items'])
	return data

def download2FileAndExtract(url, folder, fileName):
	checkFolder(folder)
	zipFileName = folder + fileName
	request = urllib2.urlopen(url)
	with open(zipFileName, "wb") as f :
		f.write(request.read())
	if not zipfile.is_zipfile(zipFileName):
		print('ERROR: ' + zipFileName + ' is not a valid zip file.')
		sys.exit(1)
	print('Extracting rotated MNIST...')
	wd = os.getcwd()
	os.chdir(folder)

	archive = zipfile.ZipFile('.'+fileName, mode='r')
	archive.extractall()
	archive.close()
	os.chdir(wd)

def download_dataset(opt):
	if opt["datasetIdx"] == 'mnist':
		print('Downloading rotated MNIST...')
		download2FileAndExtract("https://www.dropbox.com/s/0fxwai3h84dczh0/mnist_rotation_new.zip?dl=1",
			opt["data_dir"], "/mnist_rotation_new.zip")
		print('Successfully retrieved rotated MNIST dataset.')
	elif opt["datasetIdx"] == 'cifar10':
		print('Downloading CIFAR10...')
		download2FileAndExtract("https://www.dropbox.com/s/d07iifw0njuymk8/cifar_numpy.zip?dl=1",
			opt["data_dir"], "/cifar_numpy.zip")
		print('Successfully retrieved CIFAR10 dataset.')
	else:
		print('ERROR: Cannot download dataset ' + opt["datasetIdx"])
		sys.exit(1)

def load_dataset(dir_name, subdir_name, prepend=''):
	"""Loads numpy-formatted version of the datasets from the specified folder.
	We expect the numpy tensors to be named: 'trainX.npy', 'trainY.npy'
	'validX.npy', 'validY.npy', 'testX.npy', 'testY.npy'
	We check whether 'testY.npy' exists on disk and only load it if possible.
	Note: we expect images to be flattened (row-major) into 1D tensors. 

	dir_name: main data directory
	subdir_name: sub-directory for this dataset (we add a '/' between main and sub if specified)
	prepend: string to prepend to dataset numpy files as given in the description
	
	Returns dictionary with entries for 'train_x', 'train_y', 'valid_x', 'valid_y' and 'test_x'.
	If available, 'test_y' will also be loaded.
	"""
	if subdir_name != '':
		data_dir = dir_name + '/' + subdir_name
	else:
		data_dir = dir_name
	print('Loading data from directory: [ ' + data_dir + ' ]...')
	
	data = {}
	data['train_x'] = np.load(data_dir + '/' + prepend + 'trainX.npy')
	data['train_y'] = np.load(data_dir + '/' + prepend + 'trainY.npy')
	data['valid_x'] = np.load(data_dir + '/' + prepend + 'validX.npy')
	data['valid_y'] = np.load(data_dir + '/' + prepend + 'validY.npy')
	data['test_x'] = np.load(data_dir + '/' + prepend + 'testX.npy')
	if os.path.exists(data_dir + '/' + prepend + 'testY.npy'):
		data['test_y'] = np.load(data_dir + '/testY.npy')
	return data

def load_pkl(dir_name, subdir_name, prepend=''):
	"""Load dataset from subdirectory"""
	data_dir = dir_name + '/' + subdir_name
	data = {}
	with open(data_dir + '/' + prepend + 'train_images.pkl') as fp:
		data['train_x'] = pkl.load(fp)
	with open(data_dir + '/' + prepend + 'train_labels.pkl') as fp:
		data['train_y'] = pkl.load(fp)
	with open(data_dir + '/' + prepend + 'valid_images.pkl') as fp:
		data['valid_x'] = pkl.load(fp)
	with open(data_dir + '/' + prepend + 'valid_labels.pkl') as fp:
		data['valid_y'] = pkl.load(fp)
	return data

def pklbatcher(inputs, targets, batch_size, shuffle=False, augment=False,
				img_shape=(321,481,3), anneal=1.):
	"""Input and target are minibatched. Returns a generator"""
	assert len(inputs) == len(targets)
	indices = inputs.keys()
	if shuffle:
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = indices[start_idx:start_idx+batch_size]
		# Data augmentation
		im = []
		targ = []
		for i in xrange(len(excerpt)):
			img = inputs[excerpt[i]]['x']
			tg = targets[excerpt[i]]['y'] > 2
			if augment:
				# We use shuffle as a proxy for training
				if shuffle:
					img, tg = bsd_preprocess(img, tg)
			im.append(img)
			targ.append(tg)
		im = np.stack(im, axis=0)
		targ = np.stack(targ, axis=0)
		yield im, targ, excerpt

def imagenet_batcher(data, batch_size, shuffle=False, augment=False):
	"""Input and target are minibatched. Returns a generator"""
	image_dict = convert_imagenet_filelist_to_dict(data)
	indices = np.arange(len(image_dict.keys()))
	if shuffle:
		np.random.shuffle(indices)
	for start_idx in range(0, len(image_dict.keys()) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = indices[start_idx:start_idx+batch_size]
		# Data augmentation
		im = []
		targ = []
		for i in excerpt:
			img_address = image_dict[indices[i]]['x']
			img = skio.imread(img_address)
			if len(img.shape) == 2:
				img = skco.gray2rgb(img)
			tg = image_dict[indices[i]]['y']
			if augment:
				img = imagenet_preprocess(img)
			else:
				img = imagenet_global_preprocess(img)
			img = central_crop(img, (227,227,3))
			im.append(img)
			targ.append(tg)
		im = np.stack(im, axis=0)
		targ = np.stack(targ, axis=0)
		yield im, targ, excerpt

def convert_imagenet_filelist_to_dict(lines):
	image_dict = {}
	i = 0
	for line in lines:
		address, code = line.split('\t')
		code = int(code.replace('\n',''))
		image_dict[i] = {}
		image_dict[i]['x'] = '/media/daniel/HDD/ImageNet/ILSVRC2012_img_val/' + address
		image_dict[i]['y'] = code
		i += 1
	return image_dict

def minibatcher(inputs, targets, batch_size, shuffle=False, augment=False,
				img_shape=(95,95), crop_shape=10):
	"""Input and target are minibatched. Returns a generator"""
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = np.arange(batch_size) + start_idx
		# Data augmentation
		im = []
		for i in xrange(len(excerpt)):
			img = inputs[excerpt[i]]
			if augment:
				# We use shuffle as a proxy for training
				if shuffle:
					img = preprocess(img, img_shape, crop_shape)
				else:
					pass
			im.append(img)
		im = np.stack(im, axis=0)
		yield im, targets[excerpt]

def preprocess(im, im_shape, cm):
	'''Data normalizations and augmentations: cm cropmargin'''
	im = np.reshape(im, im_shape)
	# Random fliplr
	if (np.random.rand() > 0.5):
		im = np.fliplr(im)
	# Random crop
	im = np.pad(im, ((cm,cm),(cm,cm), (0,0)), 'constant')
	a = np.random.randint(2*cm)
	b = np.random.randint(2*cm)
	im = im[a:a+im_shape[0],b:b+im_shape[1]]
	return np.reshape(im, [np.prod(im_shape),])
	#new_angle = uniform_rand(-np.pi, np.pi)
	#im = sktr.rotate(im, new_angle)
	#new_shape = np.asarray(im_shape) - 2.*np.asarray((crop_margin,)*2)
	#return np.reshape(im, [np.prod(new_shape),])

def imagenet_global_preprocess(im):
	# Resize
	im = sktr.resize(im, (256,256))
	return ZMUV(im)

def ZMUV(im):
	im = im - np.mean(im, axis=(0,1))
	im = im / np.std(im, axis=(0,1))
	return im

def imagenet_preprocess(im):
	'''Data normalizations and augmentations'''
	# Resize
	im = sktr.resize(im, (256,256))
	# Random numbers
	fliplr = (np.random.rand() > 0.5)
	gamma = np.minimum(np.maximum(1. + np.random.randn(), 0.8), 1.2)
	angle = uniform_rand(0, 360.)
	scale = np.asarray((log_uniform_rand(1/1.1, 1.1), log_uniform_rand(1/1.1, 1.1)))
	translation = np.asarray((uniform_rand(-15,15), uniform_rand(-15,15)))
	# Flips
	if fliplr:
		im = np.fliplr(im)
	# Gamma
	im = skiex.adjust_gamma(im, gamma)
	im = ZMUV(im)
	# Affine transform (no rotation)
	affine_matrix = sktr.AffineTransform(scale=scale, translation=translation)
	im = sktr.warp(im, affine_matrix)
	# Rotate
	im = sktr.rotate(im, angle)
	return im

def central_crop(im, new_shape):
	im_shape = np.asarray(im.shape)
	new_shape = np.asarray(new_shape)
	top_left = (im_shape - new_shape)/2
	bottom_right = top_left + new_shape
	return im[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]

def uniform_rand(min_, max_):
	gap = max_ - min_
	return gap*np.random.rand() + min_

def log_uniform_rand(min_, max_, size=1):
	if size > 1:
		output = []
		for i in xrange(size):
			output.append(10**uniform_rand(np.log10(min_), np.log10(max_)))
	else:
		output = 10**uniform_rand(np.log10(min_), np.log10(max_))
	return output


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

def get_learning_rate(opt, current, best, counter, learning_rate):
	"""If have not seen accuracy improvement in delay epochs, then divide 
	learning rate by 10
	"""
	if current > best:
		best = current
		counter = 0
	elif counter > opt['delay']:
		learning_rate = learning_rate / opt['lr_div']
		counter = 0
	else:
		counter += 1
	return (best, counter, learning_rate)

#----------BSD-Specific Routines----------
def bsd_preprocess(im, tg):
	'''Data normalizations and augmentations'''
	fliplr = (np.random.rand() > 0.5)
	flipud = (np.random.rand() > 0.5)
	gamma = np.minimum(np.maximum(1. + np.random.randn(), 0.5), 1.5)
	if fliplr:
		im = np.fliplr(im)
		tg = np.fliplr(tg)
	if flipud:
		im = np.flipud(im)
		tg = np.flipud(tg)
	im = skiex.adjust_gamma(im, gamma)
	return im, tg

def bsd_save_predictions(sess, x, opt, pred, pt, data, epoch):
	"""Save predictions to output folder"""
	X = data['valid_x']
	Y = data['valid_y']
	save_path = opt['test_path'] + '/T_' + str(epoch)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	generator = pklbatcher(X, Y, opt['batch_size'], shuffle=False,
						   augment=False, img_shape=(opt['dim'], opt['dim2']))
	# Use sigmoid to map to [0,1]
	bsd_map = tf.nn.sigmoid(pred['fuse'])
	j = 0
	for batch in generator:
		batch_x, batch_y, excerpt = batch
		output = sess.run(bsd_map, feed_dict={x: batch_x, pt: False})
		for i in xrange(output.shape[0]):
			save_name = save_path + '/' + str(excerpt[i]).replace('.jpg','.png')
			im = output[i,:,:,0]
			im = (255*im).astype('uint8')
			if data['valid_x'][excerpt[i]]['transposed']:
				im = im.T
			skio.imsave(save_name, im)
			j += 1
	print('Saved predictions to: %s' % (save_path,))
