'''Shuffle files'''

import os
import random
import sys
import time


def main():
	for i in xrange(10):
		fname = '/home/dworrall/Data/ImageNet/labels/subsets/train_{:04d}.txt'.format(2**i)
		shuffled_fname = '/home/dworrall/Data/ImageNet/labels/subsets/train_s{:04d}.txt'.format(2**i)
		with open(fname, 'r') as fp:
			lines = fp.readlines()
		random.shuffle(lines)
		with open(shuffled_fname, 'w') as fp:
			fp.writelines(lines)

if __name__ == '__main__':
	main()