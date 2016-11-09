'''Copy files'''

import os
import shutil
import sys
import time

import numpy as np


def run():
	dst_dir = '/home/daniel/Data/ImageNetMini'
	with open('./files.txt', 'r') as fp:
		file_names = fp.readlines()
	for f in file_names:
		src = f.replace('\n','')
		# Get subfolder name
		subfolder = src.split('/')[-1]
		subfolder = subfolder.split('_')[0]
		dst_subfolder = dst_dir + '/' + subfolder
		if not os.path.exists(dst_subfolder):
			os.mkdir(dst_subfolder)
		dst = dst_subfolder + '/' + src.split('/')[-1]
		shutil.copy(src, dst)
		print dst


if __name__ == '__main__':
	run()