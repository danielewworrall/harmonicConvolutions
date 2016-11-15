'''Get imagenet label map'''

import os
import sys
import time

import cPickle as pkl

def dict_to_list(my_dict):
	image_dict = {}
	i = 0
	for key, val in my_dict.iteritems():
		for address in val['addr']:
			image_dict[i] = {}
			image_dict[i]['x'] = address
			image_dict[i]['y'] = val['num']
			i += 1
	return image_dict

def run():
	with open('./imagenet_categories.pkl', 'r') as fp:
		data = pkl.load(fp)
	image_dict = dict_to_list(data)
	
	mapper = {}
	for key, val in image_dict.iteritems():
		wnid = val['x'].split('/')[-1].split('_')[0]
		target = val['y']
		if wnid in mapper.keys():
			if mapper[wnid] != target:
				print('Mismatch')
		else:
			mapper[wnid] = target
	return mapper

def cnn2code():
	"""Map the CNN id to a code"""
	# Read in all file id maps
	mapper = run()
	root = '/media/daniel/HDD/ImageNet/ImageNetTxt/'
	with open(root + 'val.txt', 'r') as fp:
		val = fp.readlines()
	with open(root + 'synsets.txt') as fp:
		synsets = fp.readlines()
	
	# Build per val image map
	valid_map = {}
	with open('./valid_map.txt', 'w') as fp:
		for line in val:
			address, code = line.split(' ')
			code = code.replace('\n','')
			wnid = synsets[int(code)].replace('\n','')
			valid_map[address] = mapper[wnid]
			fp.write("%s\t%i\n" % (address, mapper[wnid]))


if __name__ == '__main__':
	#run()
	cnn2code()