import os
import numpy as np
import scipy.ndimage.interpolation as sciint
import skimage.color as skco
import skimage.exposure as skiex
import skimage.io as skio
import skimage.transform as sktr

##### HELPERS #####
def checkFolder(dir):
	"""Checks if a folder exists and creates it if not.
	dir: directory
	Returns nothing
	"""
	if not os.path.exists(dir):
		os.makedirs(dir)


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


##### CUSTOM FUNCTIONS FOR MAIN SCRIPT #####
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

def preprocess(im, im_shape, crop_margin):
	'''Data normalizations and augmentations'''
	# Random fliplr
	im = np.reshape(im, im_shape)
	new_angle = uniform_rand(-np.pi, np.pi)
	im = sktr.rotate(im, new_angle)
	new_shape = np.asarray(im_shape) - 2.*np.asarray((crop_margin,)*2)
	return np.reshape(im, [np.prod(new_shape),])

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