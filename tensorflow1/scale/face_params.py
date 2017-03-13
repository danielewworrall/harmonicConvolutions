'''Face parameters'''

import os
import random
import sys
import time


def main():
	#read_in_images('/home/dworrall/Data/faces15')
	#renumber_list('/home/dworrall/Data/faces15')
	#renumber_images('/home/dworrall/Data/faces15')
	#edit_file('/home/dworrall/Data/faces15')
	param_files('/home/dworrall/Data/faces15')


def read_in_images(folder):
	print('Reading in images')
	addresses = []
	for root, dirs, files in os.walk('{:s}/images'.format(folder)):
		for f in files:
			if '.png' in f:
				addresses.append('{:s}/{:s}'.format(root, f))
	
	random.shuffle(addresses)
	with open('{:s}/face_list.txt'.format(folder), 'w') as fp:
		for line in addresses:
			fp.write('{:s}\n'.format(line))


def renumber_list(folder):
	print('Getting numbers for each image in list')
	
	# Read in images
	with open('{:s}/face_list.txt'.format(folder), 'r') as fp:
		addresses = fp.readlines()
	images = {}
	
	# Strip parameters
	for address in addresses:
		dirname = os.path.dirname(address)
		basename = os.path.basename(address).replace('.png\n','')
		params = basename.split('_')
		
		if dirname not in images:
			images[dirname] = []
		num = len(images[dirname])
		new_name = '{:s}/face_{:03d}.png'.format(dirname,num)
		images[dirname].append((new_name,address.replace('\n',''),params))
	
	with open('{:s}/face_list_params.txt'.format(folder), 'w') as fp:
		for key, val in images.iteritems():
			for v in val:
				fp.write('{:s} {:s} {:03d} {:03d} {:03d} {:03d}\n'.format(v[0],
					v[1], int(v[2][0]), int(v[2][1]), int(v[2][2]), int(v[2][3])))


def renumber_images(folder):
	print('Renumbering each image in list')
	
	# Read in images
	with open('{:s}/face_list_params.txt'.format(folder), 'r') as fp:
		lines = fp.readlines()
		for i, line in enumerate(lines):
			line = line.replace('\n','')
			elements = line.split(' ')
			os.rename(elements[1], elements[0])
			sys.stdout.write('{:f}\r'.format((1.*i)/len(lines)))
			sys.stdout.flush()


def edit_file(folder):
	print('Editing: {:s}/face_list_params.txt'.format(folder))
	
	# Read in images
	with open('{:s}/face_list_params.txt'.format(folder), 'r') as fp:
		lines = fp.readlines()
	random.shuffle(lines)
	
	# Chunk up files
	os.mkdir('{:s}/addresses'.format(folder))
	for i in xrange(0, len(lines), 1000):
		with open('{:s}/addresses/addresses_{:03d}.txt'.format(folder, i/1000), 'w') as fp:
			for line in lines[i:i+1000]:
				line = line.replace('\n','')
				sl = line.split(' ')
				fp.write('{:s},{:03d},{:03d},{:03d},{:03d}\n'.format(sl[0],
							int(sl[2]), int(sl[3]), int(sl[4]), int(sl[5])))


def param_files(folder):
	print('Creating param files')
	
	# Read in images
	with open('{:s}/face_list_params.txt'.format(folder), 'r') as fp:
		lines = fp.readlines()
	os.mkdir('{:s}/params'.format(folder))
	
	# Chunk up files
	for line in lines:
		line = line.replace('images','params')
		line = line.replace('png','txt')
		sl = line.split(' ')
		dirname = os.path.dirname(sl[0])
		if not os.path.exists(dirname):
			os.mkdir(dirname)
		with open(sl[0], 'w') as fp:
			fp.write('{:s},{:s},{:s},{:s}'.format(sl[2],sl[3],sl[4],sl[5]))


if __name__ == '__main__':
	main()













































