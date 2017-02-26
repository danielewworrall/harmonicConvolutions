'''Generate ImageNet subsamples'''

import os
import sys
import time


def main():
	folder = '/home/dworrall/Data/ImageNet/labels/'
	get_stats(folder)
	wnid_dict = wnid_list(folder, save=True)
	for i in xrange(10):
		print 2**i
		build_train_set(folder, 2**i, wnid_dict)
	build_validation_set(folder, wnid_dict)


def read_categories(fname):
	"""Build training set, using first n_images per category"""
	categories = {}
	
	with open(fname, 'r') as fp:
		lines = fp.readlines()
	for line in lines:
		line_ = line.split('/')
		if line_[1] not in categories:
			categories[line_[1]] = []
			categories[line_[1]].append(line)
		else:
			categories[line_[1]].append(line)
	
	return categories


def build_train_set(folder, n_images, wnid_dict):
	"""Build training set, using first n_images per category"""
	fname = folder + 'train.txt'
	new_fname = folder + 'subsets/train_{:04d}.txt'.format(n_images)
	
	categories = read_categories(fname)
	with open(new_fname, 'w') as fp:
		for k, v in categories.iteritems():
			string = ['{:s},{:d}\n'.format(address.replace('\n',''),wnid_dict[k]) \
						 for address in v[:n_images]]
			fp.writelines(string)


def build_validation_set(folder, wnid_dict):
	"""Build training set, using first n_images per category"""
	fname = folder + 'validation.txt'
	new_fname = folder + 'subsets/validation.txt'
	
	with open(fname, 'r') as fp:
		lines = fp.readlines()
	with open(new_fname, 'w') as fp:
		for line in lines:
			string = line.split(' ')
			wnid = string[1].replace('\n','')
			fp.write('{:s},{:d}\n'.format(string[0],wnid_dict[wnid]))


def wnid_list(folder, save=True):
	"""Convert wnid to a number"""
	fname = folder + 'train.txt'
	new_fname = folder + 'wnid_list.txt'
	
	categories = read_categories(fname)
	wnid_lookup = [(i,cat+'\n') for i, cat in enumerate(categories.keys())]
	if save:
		with open(new_fname, 'w') as fp:
			fp.writelines(['{:d},{:s}'.format(i,s) for i, s in wnid_lookup])
	
	wnid_dict = {}
	for i, s in wnid_lookup:
		wnid_dict[s.replace('\n','')] = i
	return wnid_dict


def get_stats(folder):
	"""Get key stats about the data"""
	fname = folder + 'train.txt'
	categories = {}
	
	with open(fname, 'r') as fp:
		lines = fp.readlines()
	for line in lines:
		line = line.split('/')
		if line[1] not in categories:
			categories[line[1]] = 1
		else:
			categories[line[1]] += 1
	n_categories = len(categories)
	print min(categories.values())
	print max(categories.values())


if __name__ == '__main__':
	main()