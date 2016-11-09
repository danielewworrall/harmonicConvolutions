'''Run canny'''

import os
import sys
import time

import cv2
import skimage.io as skio

from matplotlib import pyplot as plt

def run():
	data_dir = '/home/daniel/Data/ImageNetMini'
	new_dir = '/home/daniel/Data/EdgeNetMini'
	j = 0
	for root, dirs, files in os.walk(data_dir):
		for f in files:
			if '.JPEG' in f:
				file_name = root + '/' + f
				subfolder = file_name.split('/')[-1].split('_')[0]
				im = skio.imread(file_name)
				edges = cv2.Canny(im, 0., 255.)
				new_subfolder = new_dir + '/' + subfolder
				if not os.path.exists(new_subfolder):
					os.mkdir(new_subfolder)
				new_file = new_subfolder + '/' + f
				skio.imsave(new_file, edges)
				sys.stdout.write("%i\r" %(j,))
				sys.stdout.flush()
				j += 1

if __name__ == '__main__':
	run()