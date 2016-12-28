'''Parse imagenet validation file'''

import os
import sys

import xml.etree.ElementTree as ET

def main():
	val = '/media/daniel/SAMSUNG/ImageNet/validation.txt'
	valfp = open(val, 'w')
	folder = '/media/daniel/SAMSUNG/ImageNet/val/'
	for root, dirs, files in os.walk(folder):
		for f in files:
			filename = root + f
			tree = ET.parse(filename)
			rt = tree.getroot()
			i = 0
			for child in rt.findall('object'):
				pass
			i +=1
			name = child.find('name').text
			valfp.writelines("%s %s\n" %(f.replace('.xml', '.JPEG'), name))
	valfp.close()

if __name__ == '__main__':
	main()