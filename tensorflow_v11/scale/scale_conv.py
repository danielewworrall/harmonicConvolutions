'''Scale conv'''

import os
import sys
import time
sys.path.append('../')

import cv2
import numpy as np
import tensorflow as tf
import skimage.feature as skfe
import skimage.io as skio

from scipy.ndimage.filters import gaussian_filter

import harmonic_network_lite as hn_lite
from harmonic_network_ops import *
from matplotlib import pyplot as plt


def zoom(N, image, factor):
	"""Zoom in on the center of the patch"""
	new_size = (int(factor*image.shape[0]), int(factor*image.shape[1]))
	image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
	new_coords = (int(image.shape[0]*0.5 - (N/2)), int(image.shape[1]*0.5 - (N/2)))
	
	return image[new_coords[0]:new_coords[0]+N, new_coords[1]:new_coords[1]+N]


def s_conv(X, shape, in_range, out_range, padding='VALID', name='Sc'):
	# Compute number of orientations
	c0 = 1.
	alpha = 1.1
	n_samples = np.floor((np.log(shape[0]/2) - np.log(c0)) / np.log(alpha))
	radii = c0*np.power(alpha, np.arange(n_samples))
	n_orientations = np.ceil(np.pi*radii[-1])
	# Instantiate the convolutional parameters
	R = get_scale_weights_dict(shape, out_range, 0.4, n_orientations,
		name=name+'_S',device='/gpu:0')
	P = get_phase_dict(shape[2], shape[3], out_range, name=name+'_b',
							 device='/gpu:0')
	W = get_scale_filters(R, shape[0], P=P)
	# The convolution
	return h_range_conv(X, W, in_range=in_range, out_range=out_range,
							  padding=padding, name=name+'_N')


def non_max_suppression(image, size=5):
	"""Stupid non-max suppression. Build boxes and preserve points with highest
	value."""
	from skimage.feature import peak_local_max
	return peak_local_max(image, min_distance=size)


def main():
	"""Run shallow scale conv"""
	fs = 9
	nc = 10
	
	X = skio.imread('../images/balloons.jpg')[50:250,150:350,:]
	x = tf.placeholder(tf.float32, [1,200,200,1,1,3], name='x')
	
	# The convolutions
	s1 = s_conv(x, [fs,fs,3,nc], (0,0), (0,2), name='sc1')
	s1 = hn_lite.nonlinearity(s1, fnc=tf.nn.relu, name='nl1', device='/gpu:0')
	
	s2 = s_conv(s1, [fs,fs,nc,nc], (0,2), (0,2), name='sc2')
	s2 = hn_lite.nonlinearity(s2, fnc=tf.nn.relu, name='nl2', device='/gpu:0')
	
	y = s_conv(s2, [fs,fs,nc,1], (0,2), (0,2), name='sc3')
	mag = stack_magnitudes(y)
	Mags = []
	
	scales = 1 + np.arange(100)*0.04
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		for i in xrange(len(scales)):
			X_zoom = zoom(200, X, scales[i])
			X_ = X_zoom[np.newaxis,:,:,np.newaxis,np.newaxis,:]
			Mag = sess.run(mag, feed_dict={x: X_})
			Mags.append(Mag)
	
	rescaled = []
	for i, m in enumerate(Mags):
		im = np.squeeze(m)[:,:,0]
		factor = 1./scales[i]
		new_size = (int(factor*im.shape[0]), int(factor*im.shape[1]))
		im = cv2.resize(im, new_size, interpolation=cv2.INTER_CUBIC)
		rescaled.append(im)
		
	#plt.imshow(rescaled[0], interpolation='nearest')
	coords = non_max_suppression(rescaled[0], 3)
	#plt.scatter(coords[:,1],coords[:,0], color='g')
	#plt.show()	
	
	plt.ion()
	plt.show()
	# Repeatability
	sh = rescaled[0].shape
	MSE = []
	for i in xrange(len(rescaled)-1):
		print scales[i]
		shape = rescaled[i+1].shape
		
		#cnew.append(coords[:,1]+(-shape[0]+sh[0])/2,coords[:,0]+(-shape[1]+sh[1])/2)
		
		
		left = ((sh[0]-shape[0])/2, (sh[1]-shape[1])/2)
		right = (left[0]+shape[0], left[1]+shape[1])
		crop = rescaled[0][left[0]:right[0], left[1]:right[1]]
		
		coords = non_max_suppression(rescaled[i+1], 3)
		
		plt.cla()
		plt.imshow(rescaled[0], interpolation='nearest', cmap='gray')
		plt.scatter(coords[:,1]+(-shape[0]+sh[0])/2,coords[:,0]+(-shape[1]+sh[1])/2, color='r')
		plt.xlim(0,200)
		plt.ylim(0,200)
		plt.draw()
		raw_input(i)
		
def tests():
	"""Benchmark comparisons"""
	
	fs = 9
	nc = 10
	ni = 1
	x = tf.placeholder(tf.float32, [1,680,850,1,1,ni], name='x')
	#s1 = s_conv(x, [fs,fs,ni,nc], (0,0), (0,2), padding='SAME', name='sc1')
	s1 = hn_lite.conv2d(x, nc, 7, padding='SAME', name='lc1', device='/gpu:0')
	s1 = hn_lite.nonlinearity(s1, fnc=tf.nn.relu, name='nl1', device='/gpu:0')
	#s2 = s_conv(s1, [fs,fs,nc,nc], (0,2), (0,2), padding='SAME', name='sc2')
	#s2 = hn_lite.conv2d(s1, nc, 7, padding='SAME', name='lc2', device='/gpu:0')
	#s2 = hn_lite.nonlinearity(s2, fnc=tf.nn.relu, name='nl2', device='/gpu:0')
	#y = s_conv(s2, [fs,fs,nc,1], (0,2), (0,2), padding='SAME', name='sc3')
	y = hn_lite.conv2d(s1, 1, 7, padding='SAME', name='lc3', device='/gpu:0')
	mag = stack_magnitudes(y)
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		plt.ion()
		plt.show()
		coords = []		
		for i in xrange(6):
			print i
			tag = 'img'+str(i+1)
			X = skio.imread('../../../Desktop/oxford/boat/'+tag+'.pgm')/255.
			print np.amin(X), np.amax(X)
			# Pass image through network
			X_ = np.reshape(X, (1,680,850,1,1,1))
			Mag = sess.run(mag, feed_dict={x: X_})
			Mag = np.squeeze(Mag)[:,:,0]/100.
			print np.amin(Mag), np.amax(Mag)
			# Homography output and pick keypoints
			if i > 0:
				address = '../../../Desktop/oxford/boat/H1to'+str(i+1)+'p'
				#Mag = homography(Mag, address)
			#Mag = gaussian_filter(Mag, 5.)
			
			coords.append(non_max_suppression(Mag, 5))
			print coords[-1].shape
			write_coords_to_file(coords[-1], './coords/img'+str(i+1)+'.hnetaf')
			
			# Harris corner detector
			#affine_region_detector(X, i)
			
			plt.figure(1)
			plt.cla()
			plt.imshow(Mag)
			X_ = X
			if i > 0:
				address = '../../../Desktop/oxford/boat/H1to'+str(i+1)+'p'
				X_ = homography(X_, address)
			plt.figure(2)
			plt.imshow(X_, interpolation='nearest', cmap='gray')
			print np.sqrt(np.mean(X_-Mag)**2)
			#plt.scatter(coords[-1][:,1], coords[-1][:,0], color='r')
			plt.draw()
			raw_input()
		
		for j in xrange(5):
			D = nearest_neighbour(coords[0], coords[j+1])
			print np.mean(D < 1), np.mean(D < 2), np.mean(D < 5), np.mean(D < 10)
			

def affine_region_detector(X, i):
	im = skfe.corner_harris(X)
	im = gaussian_filter(im, 2.)
	ch = non_max_suppression(im, 5)
	
	Hrr, Hrc, Hcc = skfe.hessian_matrix(X)
	hrr = Hrr[ch[:,0],ch[:,1]]
	hrc = Hrc[ch[:,0],ch[:,1]]
	hcc = Hcc[ch[:,0],ch[:,1]]
	write_affine_coords_to_file(ch, './coords/img'+str(i+1)+'.hnetaf', hrr, hrc, hcc)


def homography(X, address):
	import skimage.transform as sktr
	
	with open(address, 'r') as fp:
		lines = fp.readlines()
		H = []
		for line in lines:
			line = line.replace('\n','')
			line = line.split()
			row = [float(i) for i in line]
			H.append(np.hstack(row))
		H = np.vstack(H)
	im = sktr.warp(X, H)
	return im



def nearest_neighbour(X, Y):
	"""Assume X and Y are column matrices. Find distance
	D = sqrt((X0-Y0)**2 + (X1-Y1)**2) between all pairs of points, then minimize
	per row/column"""
	X = X[np.newaxis,:,:]
	Y = Y[:,np.newaxis,:]
	D = np.sqrt((X[:,:,0]-Y[:,:,0])**2 + (X[:,:,1]-Y[:,:,1])**2)
	return np.amin(D, axis=1)


def write_coords_to_file(coords, address):
	with open(address, 'w') as fp:
		fp.writelines('1.0\n')
		fp.writelines(str(coords.shape[0])+'\n')
		for coord in coords:
			string = str(float(coord[0]))+'  '+str(float(coord[1]))+'  0.005  0.0  0.005\n'
			fp.writelines(string)


def write_affine_coords_to_file(coords, address, A, B, C):
	with open(address, 'w') as fp:
		fp.writelines('1.0\n')
		fp.writelines(str(coords.shape[0])+'\n')
		for coord, a, b, c in zip(coords, A, B, C):
			a = 0.005
			b = 0.0
			c = 0.005
			string = str(float(coord[0])) + '  ' + str(float(coord[1]))
			string += '  ' + str(a) + '  ' + str(b) + '  ' + str(c) + '\n'
			fp.writelines(string)


if __name__ == '__main__':
	#main()
	tests()





































