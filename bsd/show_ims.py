import os

import skimage.io as skio
from matplotlib import pyplot as plt


plt.ion()
plt.show()
for root, dirs, files in os.walk('.'):
	for f in files:
		fname = root + '/' + f
		im = skio.imread(fname)
		plt.imshow(im, interpolation='nearest', cmap='gray')
		plt.draw()
		raw_input(f)
