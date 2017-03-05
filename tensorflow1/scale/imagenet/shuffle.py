'''Shuffle files'''

import os
import random
import sys
import time


def main():
	fname = '/home/dworrall/Data/ImageNet/labels/top_k/train_0050.txt'
	shuffled_fname = '/home/dworrall/Data/ImageNet/labels/top_k/train_s0050.txt'
	with open(fname, 'r') as fp:
		lines = fp.readlines()
	random.shuffle(lines)
	with open(shuffled_fname, 'w') as fp:
		fp.writelines(lines)

if __name__ == '__main__':
	main()