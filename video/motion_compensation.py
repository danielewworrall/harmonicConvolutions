'''Motion compensation'''

import os
import sys
import time

import numpy as np
import skimage.color as skco
import skimage.draw as skdr
import skimage.io as skio
import skimage.transform as sktr

from matplotlib import pyplot as plt

def run():
	data_dir = './rot/cnn_rot_conv1_2'
	save_dir = './mot_comp'
	
	for i, angle in enumerate(np.linspace(0., 360., num=721)):
		im_name = data_dir + '/im_' + '{:04d}'.format(i) + '.png'
		im = skio.imread(im_name)
		#offset = int(100.*np.sin(2*np.pi*angle/360.))
		#im = draw_box(im, 200, 200, offset, lw=3)
		im_ = sktr.rotate(im, -angle)
		
		im = im.astype('uint16')
		save_name = save_dir + '/im_' + '{:04d}'.format(i) + '.png'
		print save_name
		skio.imsave(save_name, im_)
		#plt.imshow(im[50:150,50-offset:150-offset], interpolation='nearest')
		#plt.imshow(im, interpolation='nearest', cmap='gray')
		##plt.draw()
		#raw_input(i)

def save_box():
	data_dir = './trans/hnet_trans_conv1_2'
	save_dir = './trans/'
	
	im_name = data_dir + '/im_0000.png'
	im = skio.imread(im_name)
	im = cut_box(im, 200, 200)

	save_name = save_dir + '/hnet_trans_conv1_cut.png'
	skio.imsave(save_name, im)

def draw_box(im, width, height, offset, lw=5):
	imsh = im.shape
	box = np.ones(imsh)
	left = int(imsh[1]/2.) - width/2 - offset
	right = left + width
	top = int(imsh[0]/2.) - height/2
	bottom = top + height
	box[top:bottom, left:right] = 0.
	box[top+lw:bottom-lw, left+lw:right-lw] = 1.
	return im * box + (1.-box)*np.amax(im)

def cut_box(im, width, height):
	imsh = im.shape
	box = np.ones(imsh)
	left = int(imsh[1]/2.) - width/2 
	right = left + width
	top = int(imsh[0]/2.) - height/2
	bottom = top + height
	
	return im [top:bottom, left:right]


if __name__ == '__main__':
	run()
	#save_box()