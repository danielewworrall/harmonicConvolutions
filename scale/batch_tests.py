'''Batch tests'''

import os
import time
import sys

import numpy as np

import sgd_equivariance as se


def main():
	opt = {}
	opt['equivariant_weight'] = 0. #1e-3
	accs = []
	N_trials = 10
	'''
	for i in xrange(N_trials):
		acc = se.main(opt)
		accs.append(acc)
		np.save('./batch_tests/equi_0b.npy', accs)
	'''
	data0 = np.load('./batch_tests/equi_0b.npy')
	data3 = np.load('./batch_tests/equi_1e_n3b.npy')
	print data0
	print('Plain: {:04f}, {:04f}'.format(np.mean(data0), np.std(data0)))
	print data3
	print('Equi: {:04f}, {:04f}'.format(np.mean(data3), np.std(data3)))
	


if __name__ == '__main__':
	main()
