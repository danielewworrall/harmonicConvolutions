'''re batch cifar'''

import os
import sys
import time

import numpy as np

def rebatch():
	train = np.load('../data/cifar10/preprocessed/train.npz')
	valid = np.load('../data/cifar10/preprocessed/val.npz')
	test = np.load('../data/cifar10/preprocessed/test.npz')
	dir_name = '../data/cifar10/preprocessed'
	np.save(dir_name+'/trainX.npy', train['data'])
	np.save(dir_name+'/trainY.npy', train['labels'])
	np.save(dir_name+'/validX.npy', valid['data'])
	np.save(dir_name+'/validY.npy', valid['labels'])
	np.save(dir_name+'/testX.npy', test['data'])
	np.save(dir_name+'/testY.npy', test['labels'])


if __name__ == '__main__':
	rebatch()