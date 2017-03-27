'''Get MultiPIE addresses'''

import os
import random
import sys
import time


def readin_images(folder):
	'''Get all addresses of all files in folder'''
	addresses = []
	for root, dirs, files in os.walk(folder):
		for f in files:
			if '.png' in f:
				addresses.append('{:s}/{:s}'.format(root, f))
	
	# In place random shuffle of the addresses
	random.shuffle(addresses)
	with open('{:s}/addresses.txt'.format(folder)):
		for address in addresses:
			stats = address.replace('.png').split('_')
			fp.write('{:s},{:s},{:s},{:s},{:s},{:s}\n'.format(address,stats[0],stats[1],stats[2],stats[3],stats[4]))



if __name__ == '__main__':
	folder = '/home/dworrall/Data/multipie'
	readin_images(folder)