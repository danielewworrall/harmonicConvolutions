'''ImageNet Mini Classification List'''

import os
import sys
import time

import cPickle as pkl

def run():
	data_dir = '/home/sgarbin/data/ImageNetMini'
	images = {}
	i = 0
	for root, dirs, files in os.walk(data_dir):
		for f in files:
			file_name = root + '/' + f
			if '.JPEG' in f:
				category = f.split('_')[0]
				if category not in images:
					images[category] = {}
					images[category]['num'] = i
					i += 1
					images[category]['addr'] = []
					print category
				images[category]['addr'].append(file_name)
	
	print('%i categories' % (len(images.keys()),))
	with open('./imagenet_categories.pkl', 'w') as fp:
		pkl.dump(images, fp, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	run()