'''Run canny'''

import os
import sys
import time

import cv2
import numpy as np
import skimage.color as skco
import skimage.io as skio
import skimage.transform as sktr

from matplotlib import pyplot as plt

def run():
	data_dir = '/home/sgarbin/data/ImageNetMini'
	j = 0
	for root, dirs, files in os.walk(data_dir):
		for f in files:
			if '.JPEG' in f:
				file_name = root + '/' + f
				subfolder = file_name.split('/')[-1].split('_')[0]
				im = skio.imread(file_name)
				im_size = np.asarray(im.shape)[:2]
				new_size = np.floor(im_size * (481./ np.amax(im_size))).astype(np.int)
				im = sktr.resize(im, new_size)
				if len(im_size) == 2:
					im = skco.gray2rgb(im)
				if new_size[0] > new_size[1]:
					im = np.transpose(im, (1,0,2))
				skio.imsave(file_name, im)
				sys.stdout.write("%i\r" %(j,))
				sys.stdout.flush()
				j += 1

if __name__ == '__main__':
	run()
