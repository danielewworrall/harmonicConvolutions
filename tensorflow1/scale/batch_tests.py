'''Batch tests'''

import os
import time
import sys

import numpy as np
import seaborn as sns

import mnist_equivariance as se

from matplotlib import pyplot as plt
sns.set_style("whitegrid")

def main():

	opt = {}
	for equi_weight in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.]:
		opt['equivariant_weight'] = equi_weight
		accs = []
		N_trials = 10
		
		for i in xrange(N_trials):
			acc = se.main(opt)
			accs.append(acc)
			np.save('./batch_tests/equi_{:.0e}.npy'.format(opt['equivariant_weight']), accs)
		
	'''
	plt.figure(1)
	equi_weights = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
	
	threshold = 0.
	
	means = []
	stddevs = []
	for equi_weight in equi_weights:
		print equi_weight
		fname = './batch_tests/equi_{:.0e}.npy'.format(equi_weight)
		data = np.load(fname)
		data = data[data > threshold]
		means.append(np.mean(data))
		stddevs.append(np.std(data))
		print data
		print('{:s}: {:04f}, {:04f}'.format(fname, 1-means[-1], stddevs[-1]))
	
	# No equivariance
	fname = './batch_tests/equi_0e+00.npy'.format(equi_weight)
	data0 = np.load(fname)
	data0 = data0[data0 > threshold]
	mean0 = np.mean(data0)
	stddev0 = np.std(data0)
	print data0
	print('{:s}: {:04f}, {:04f}'.format(fname, 1-mean0, stddev0))
		
	means = 100.*(1.-np.asarray(means))
	stddevs = 100.*np.asarray(stddevs)
	plt.errorbar(equi_weights, means, yerr=stddevs)
	plt.plot([0.9*1e-6,1.1*1e-1], [100.*(1.-mean0), 100.*(1.-mean0)], 'r-')
	plt.plot([0.9*1e-6,1.1*1e-1], [100.*(1.-mean0-stddev0), 100.*(1.-mean0-stddev0)], 'r--')
	plt.plot([0.9*1e-6,1.1*1e-1], [100.*(1.-mean0+stddev0), 100.*(1.-mean0+stddev0)], 'r--')
	plt.xscale('log')
	plt.xlabel('Equivariance weight (log scale)', fontsize=16)
	plt.ylabel('Test error %', fontsize=16)
	plt.xlim(0.9*1e-6,1.1*1e-1)
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tight_layout()
	plt.show()
	'''
	

if __name__ == '__main__':
	main()
