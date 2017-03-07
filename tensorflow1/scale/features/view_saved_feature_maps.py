'''View rotated feature maps'''

import os
import sys
import time
sys.path.append('../')

import cv2
import numpy as np
import scipy.ndimage.interpolation as spint
import skimage.io as skio
import skimage.transform as sktr

from matplotlib import pyplot as plt


def load_feature_maps(folder):
	"""Load all features from a certain layer"""
	features = {}
	for root, dirs, files in os.walk(folder):
		for f in files:
			key = f.split('_')[1].replace('.npz','')
			features[float(key)] = np.load('{:s}/{:s}'.format(root,f))
	return features


def return_sorted_features(features, indices):
	"""Return the features, sorted and from a certain layer"""
	my_list = []
	angles = []
	for key in sorted(features):
		feature_map = features[key]['arr_0']
		my_list.append(feature_map[indices])
		angles.append(key)
	return my_list, angles


def display(features, angles, folder):
	"""Display feature maps"""
	# Need to extract data value limits for normalization
	min_ = np.inf
	max_ = -np.inf
	for feature in features:
		min_ = min(np.amin(feature), min_)
		max_ = max(np.amax(feature), max_)
	for i, feature in enumerate(features):
		fname = '{:s}/image_{:04d}.png'.format(folder, i)
		feat0 = 0.95*feature[...,0]/(max_-min_)
		feat1 = 0.95*feature[...,1]/(max_-min_)
		feat_mag = np.sqrt(feat0**2 + feat1**2)
		# Unrotate images
		feat = motion_correction(feat_mag, -360.*angles[i]/(2.*np.pi))
		skio.imsave(fname, feat)


def motion_correction(image, angle):
	"""Motion correction: rotations for now"""
	return sktr.rotate(image, angle, order=1, preserve_range=True)


def rotateImage(image, angle):
  """Motion correction"""
  return spint.rotate(image, angle, order=1, reshape=False)

'''
def stability(features, angles):
	"""Display feature maps"""
	# Need to extract data value limits for normalization	
	corrected_features = []
	for i, feature in enumerate(features):
		feat0 = feature[...,0]
		feat1 = feature[...,1]
		feat_mag = np.sqrt(feat0**2 + feat1**2)
		# Unrotate images
		corrected_feature = motion_correction(feat_mag, -360.*angles[i]/(2.*np.pi))
		corrected_features.append(corrected_feature)
	
	errors = []
	for cf in corrected_features:
		# Center crop the images to ignore edge effects
		feat0 = corrected_features[0][28:84,28:84]
		feat1 = cf[28:84,28:84]
		error = feat0 - feat1
		error2 = np.mean(error**2)
		# Normalize
		error2 /= np.sqrt(np.mean(corrected_features[0]**2)) #*np.sqrt(np.mean(cf**2))
		errors.append(error2)
	return np.hstack(errors)
'''

def block_stability(features, angles):
	"""Return invariant feature maps for blocks"""
	# Need to extract data value limits for normalization	
	corrected_features = []
	for i, feature in enumerate(features):
		feat0 = feature[0,...,::2]
		feat1 = feature[0,...,1::2]
		feat_mag = np.sqrt(feat0**2 + feat1**2)
		# Unrotate images
		corrected_feature = rotateImage(feat_mag, -360.*angles[i]/(2.*np.pi))
		corrected_features.append(corrected_feature)
		sys.stdout.write('{:d}\r'.format(i))
		sys.stdout.flush()
	
	errors = []
	lo = int(0.25*corrected_features[0].shape[0])
	hi = int(0.75*corrected_features[0].shape[0])
	for cf in corrected_features:
		# Center crop the images to ignore edge effects
		feat0 = corrected_features[0][lo:hi,lo:hi]
		feat1 = cf[lo:hi,lo:hi]
		#error = feat0 - feat1
		#error2 = np.mean(error**2)
		# Normalize
		#error2 /= np.mean(corrected_features[0]**2) #*np.sqrt(np.mean(cf**2))
		#error = np.sqrt(error2)
		error = np.sum(feat0*feat1) / (np.sqrt(np.sum(feat0**2))*np.sqrt(np.sum(feat1**2)))
		errors.append(error)
	return np.hstack(errors)

'''
def multi_network_stability():
	"""Plot stability across multiple networks"""
	layer_num = 0
	slice_ = np.s_[0,:,:,0:2]
	
	fig_handle = {}
	for equi_weight in [1e-3,1e-2,1e-1,1e+0,1e+1]:
		folder = './feature_maps/feature_maps_{:.0e}/layer_{:d}'.format(equi_weight, layer_num)
		features = load_feature_maps(folder)
		maps, angles = return_sorted_features(features, slice_)
		errors = stability(maps, angles)
		plt.plot(errors, label='{:1.0e}'.format(equi_weight))
		
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.xlabel('Angle (deg)', fontsize=16)
	plt.ylabel('Per-pixel normalized RMSE', fontsize=16)
	plt.legend(fontsize=16)
	plt.tight_layout()
	plt.xlim([-5,365])
	plt.show()
'''

def multi_network_all_fms_stability():
	"""Plot stability across multiple networks for entire feature maps"""
	layer_num = 0
	slice_ = np.s_[...]
	
	fig_handle = {}
	for equi_weight in [1e-3,1e-2,1e-1,1e+0,1e+1]:
		print('Equi weight: {:f}'.format(equi_weight))
		folder = './feature_maps/feature_maps_{:.0e}/layer_{:d}'.format(equi_weight, layer_num)
		features = load_feature_maps(folder)
		maps, angles = return_sorted_features(features, slice_)
		errors = block_stability(maps, angles)
		plt.plot(errors, label='{:1.0e}'.format(equi_weight))
		
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.xlabel('Angle (deg)', fontsize=16)
	plt.ylabel('Per-pixel normalized inner product', fontsize=16)
	plt.legend(fontsize=16)
	plt.tight_layout()
	plt.xlim([-5,365])
	plt.show()



def main():
	folder = './feature_maps/layer_0'
	save_folder = './images'
	features = load_feature_maps(folder)
	maps, angles = return_sorted_features(features, np.s_[0,:,:,0:2])
	#display(maps, angles, save_folder)
	stability(maps, angles)


if __name__ == '__main__':
	#multi_network_stability()
	multi_network_all_fms_stability()






























