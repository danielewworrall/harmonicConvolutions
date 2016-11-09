'''Run canny'''

import os
import sys
import time

import cv2
import numpy as np
import skimage.io as skio
import skimage.transform as sktr

from matplotlib import pyplot as plt

def run():
	data_dir = '/home/daniel/Data/EdgeNetMini'
	j = 0
	for root, dirs, files in os.walk(data_dir):
		for f in files:
			if '.JPEG' in f:
				file_name = root + '/' + f
				subfolder = file_name.split('/')[-1].split('_')[0]
				im = skio.imread(file_name)
				im_size = np.asarray(im.shape)
				new_size = np.floor(im_size * (481./ np.amax(im_size))).astype(np.int)
				im = sktr.resize(im, new_size)
				skio.imsave(file_name, im)
				sys.stdout.write("%i\r" %(j,))
				sys.stdout.flush()
				j += 1

if __name__ == '__main__':
	run()