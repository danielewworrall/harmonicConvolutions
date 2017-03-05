'''Generate ImageNet subsamples'''

import operator
import os
import sys
import time

import numpy as np
def main():
	k = 25
	folder = '/home/dworrall/Data/ImageNet/labels/'
	get_stats(folder)
	wnid_dict = wnid_list(folder, save=False)
	#for i in xrange(10):
	#	print 2**i
	#	build_train_set(folder, 2**i, wnid_dict)
	#build_validation_set(folder, wnid_dict)
	sorted_keys, wnid_relabels = top_k_categories(folder, k, wnid_dict)
	top_k_validation(folder, k, sorted_keys, wnid_dict, wnid_relabels)


def top_k_validation(folder, k, sorted_keys, wnid_dict, wnid_relabels):
	"""Only use the validation examples from the top k sorted keys"""
	fname = folder + '/top_k/validation.txt'
	new_fname = folder + 'top_k/validation_{:04d}.txt'.format(k)
	'''
	top_k = []
	n_cats = len(sorted_keys)
	for i in xrange(k):
		key = sorted_keys[n_cats-i-1][0]
		top_k.append(wnid_dict[key])
	'''
		
	with open(fname, 'r') as fp:
		lines = fp.readlines()
	with open(new_fname, 'w') as fp:
		for line in lines:
			string = line.split(',')
			wnid_num = int(string[1].replace('\n',''))
			wnid = wnid_dict.keys()[wnid_dict.values().index(wnid_num)]
			if wnid in wnid_relabels:
				fp.write('{:s},{:d}\n'.format(string[0],wnid_relabels[wnid]))


def top_k_categories(folder, k, wnid_dict):
	"""Keep the k categories of largest size"""
	fname = folder + 'train.txt'
	new_fname = folder + 'top_k/train_{:04d}.txt'.format(k)
	relabel_fname = folder + 'top_k/relabels_{:04d}.txt'.format(k)
	
	categories = read_categories(fname)
	category_sizes = {}
	for key, val in categories.iteritems():
		category_sizes[key] = len(val)
	sorted_keys = sorted(category_sizes.items(), key=operator.itemgetter(1))
	
	n_cats = len(category_sizes)
	wnid_relabels = {}
	for j in xrange(k):
		key = sorted_keys[n_cats-j-1][0]
		wnid_relabels[key] = j
	
	with open(new_fname, 'w') as fp:
		for i in xrange(k):
			key = sorted_keys[n_cats-i-1][0]
			new_label = wnid_relabels[key]
			string = ['{:s},{:d}\n'.format(address.replace('\n',''),new_label) \
						 for address in categories[key]]
			fp.writelines(string)
	
	with open(relabel_fname, 'w') as fp:
		for key, val in wnid_relabels.iteritems():
			fp.write('{:s},{:d}\n'.format(key,val))
	return sorted_keys, wnid_relabels


def read_categories(fname):
	"""Return categories"""
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