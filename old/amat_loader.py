'''Load the .amat file format'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

def main():
	folder_name = './data/mnist_rotation_new/'
	train_name = folder_name + 'mnist_all_rotation_normalized_float_train_valid.amat'
	test_name = folder_name + 'mnist_all_rotation_normalized_float_test.amat'
	
	print('Converting amat to numpy')
	x_train, y_train = amat_to_numpy(train_name)
	x_valid, y_valid = x_train[10000:], y_train[10000:]
	x_train, y_train = x_train[:10000], y_train[:10000]
	x_test, y_test = amat_to_numpy(test_name)
	print x_train.shape, y_train.shape, x_valid.shape, y_valid.shape
	
	print('Saving data')
	np.savez(folder_name + 'rotated_train.npz', x=x_train, y=y_train)
	np.savez(folder_name + 'rotated_valid.npz', x=x_valid, y=y_valid)
	np.savez(folder_name + 'rotated_test.npz', x=x_test, y=y_test)

def amat_to_numpy(file_name):
	with open(file_name, 'r') as fp:
		lines = fp.readlines()
	
	digits = []
	labels = []
	for i in xrange(len(lines)):
		line = "".join(lines[i])
		line = line.split(' ')

		digit = np.asarray(line[:784], dtype=np.float32)
		digits.append(np.reshape(digit, [28,28]).T)
		label = line[784].split('.')[0]
		labels.append(np.int32(label))

	digits = np.reshape(np.stack(digits, axis=0), [-1, 784])
	labels = np.hstack(labels)
	return digits, labels

if __name__ == '__main__':
	main()
