'''Collect imagenet'''

import os
import sys
import time

import numpy as np
import skimage.io as skio


def run():
	data_dir = '/home/daniel/Data/ImageNet/ILSVRC2012_img_train/'
	addresses = []
	for root, dirs, files in os.walk(data_dir):
		files_to_choose = []
		for f in files:
			files_to_choose.append(root + '/' + f)
		if len(files_to_choose) > 10:
			choice = np.random.choice(len(files_to_choose), size=100)
			for c in choice:
				addresses.append(files_to_choose[c] + '\n')
	with open('./files.txt', 'w') as fp:
		fp.writelines(addresses)
	

if __name__ == '__main__':
	run()