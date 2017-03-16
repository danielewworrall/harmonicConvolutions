'''Movie strip'''

import os
import sys
import time

import numpy as np
import skimage.io as skio
import skimage.transform as sktr

from matplotlib import pyplot as plt

def movie_stripper(folder):
	img = []
	for subfolder in ['az_light','el_light','az_rot','el_rot']:
		fnames = []
		for root, dirs, files in os.walk(folder + '/' + subfolder):
			for f in files:
				if 'original' in f:
					original = root + '/' + f
				else:
					fnames.append(int(f.split('.')[0]))
		
		imgs = []
		imgs.append(sktr.resize(skio.imread(original), (300,300), order=0))

		for i in np.sort(fnames):
			if i % 1 == 0:
				fname = '{:s}/{:s}/{:04d}.png'.format(folder, subfolder, i)
				imgs.append(skio.imread(fname)/255.)
		
		img.append(np.hstack(imgs))
	img = np.vstack(img)
		
	plt.imshow(img)
	plt.axis('off')
	plt.show()


if __name__ == '__main__':
	movie_stripper('/home/dworrall/Code/harmonicConvolutions/tensorflow1/scale/validation_samples')