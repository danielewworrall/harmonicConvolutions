'''Multiresolution merge'''

import os
import sys
import time

import numpy as np
import skimage.io as skio
import skimage.transform as sktr

from matplotlib import pyplot as plt

def run():
	data_dir = './bsd/trialR/'
	epoch = 150
	# Get file_names
	dir_name = data_dir + 'S1_T_' + str(epoch)

	#plt.ion()
	#plt.show()
	for root, dirs, files in os.walk(dir_name):
		for f in files:
			if '.png' in f:
				plt.clf()
				ims = []
				for scale in [1,2,4]:
					transposed = False
					im_name = data_dir + 'S' + str(scale) + '_T_' + str(epoch) + '/' + f
					im = skio.imread(im_name)
					if im.shape[0] > im.shape[1]:
						im = im.T
						transposed = True
					ims.append(sktr.resize(im, (321, 481)))

				im = np.stack(ims, axis=2)
				im = np.mean(im, axis=2)
				if transposed:
					im = im.T
				new_name = data_dir + 'merged_' + str(epoch) + '/' + f
				skio.imsave(new_name, im)
				print f
				#plt.imshow(im, interpolation='nearest', cmap='gray')
				#plt.draw()
				#raw_input(f)
	


if __name__ == '__main__':
	run()