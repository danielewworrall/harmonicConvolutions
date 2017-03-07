'''Chunk up files'''

import os
import sys
import time


def main(N=1000):
	folder = '/home/dworrall/Data/ImageNet/labels/subsets'
	for i in [1,2,4,8,16,32,64,128,256,512]:
		fname = folder + '/train_s{:04d}.txt'.format(i)
		new_folder = folder + '/train_{:04d}'.format(i)
		os.mkdir(new_folder)
		
		with open(fname, 'r') as fp:
			lines = fp.readlines()
		for j in xrange(0,len(lines),N):
			new_fname = folder + '/train_{:04d}/chunk_{:05d}.txt'.format(i,j/N)
			with open(new_fname, 'w') as fp:
				fp.writelines(lines[j:j+N])
	'''			
	fname = folder + '/validation.txt'
	new_folder = folder + '/validation'
	os.mkdir(new_folder)
	
	with open(fname, 'r') as fp:
		lines = fp.readlines()
	for j in xrange(0,len(lines),N):
		new_fname = folder + '/validation/chunk_{:05d}.txt'.format(j/N)
		with open(new_fname, 'w') as fp:
			fp.writelines(lines[j:j+N])
	'''

if __name__ == '__main__':
	main()