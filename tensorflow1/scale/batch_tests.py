'''Batch tests'''

import os
import time
import sys

import numpy as np

import sgd_equivariance as se

from matplotlib import pyplot as plt


def main():
	opt = {}
	opt['equivariant_weight'] = 1e-6
	accs = []
	N_trials = 10
	'''
	for i in xrange(N_trials):
		acc = se.main(opt)
		accs.append(acc)
		np.save('./batch_tests/equi_{:.0e}.npy'.format(opt['equivariant_weight']), accs)
	'''
	plt.figure(1)
	equi_weights = [1e-1, 1e-2, 1e-3]
	
	means = []
	stddevs = []
	for equi_weight in equi_weights:
		fname = './batch_tests/equi_{:.0e}.npy'.format(equi_weight)
		data = np.load(fname)
		means.append(np.mean(data))
		stddevs.append(np.std(data))
		print data
		print('{:s}: {:04f}, {:04f}'.format(fname, 1-means[-1], stddevs[-1]))
	plt.errorbar(equi_weights, means, yerr=stddevs)
	plt.xscale('log')
	plt.show()
	

if __name__ == '__main__':
	main()
