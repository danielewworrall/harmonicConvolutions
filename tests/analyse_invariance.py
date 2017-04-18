'''Analyse invariance'''

import os
import sys

import numpy as np
from scipy.ndimage import rotate
from skimage.io import imread, imsave

from matplotlib import pyplot as plt

folder_name = 'nate_5_'

im0 = imread('./nate_experiments/activations/{:s}/{:04d}.jpg'.format(folder_name,0))/255.
error = []
errorm = []
for i in xrange(360):
	fname = './nate_experiments/activations/{:s}/{:04d}.jpg'.format(folder_name,i)
	im = imread(fname)/255.

	dev = np.sqrt((im - im0)**2 / (im0**2 + 1e-6))[82:401,82:401]
	error.append(np.mean(dev))

	dev= (im - im0)
	imsave('./nate_experiments/dev/{:s}/{:04d}.jpg'.format(folder_name,i), dev)

plt.plot(error, 'b')
plt.xlabel('Input angle', fontsize=20)
plt.ylabel('Mean discepancy', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.show()
'''

z = np.zeros((10,10))
z[2,2] = 1
z[1,1] = 1
z = rotate(z, 90, axes=(1,0), reshape=False, order=1)

print z
plt.imshow(z, vmin=0, vmax=1, interpolation='nearest')
plt.show()
'''