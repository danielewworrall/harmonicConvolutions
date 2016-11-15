'''Show images in a file'''

import os

import numpy as np

import skimage.io as skio
from matplotlib import pyplot as plt


plt.ion()
plt.show()
for root, dirs, files in os.walk('./trialA/T_45'):
	for f in files:
		fname = root + '/' + f
		im = skio.imread(fname)
		print im.dtype
		print np.amin(im), np.amax(im), np.mean(im)
		plt.imshow(im, interpolation='nearest', cmap='gray')
		plt.draw()
		raw_input(f)
