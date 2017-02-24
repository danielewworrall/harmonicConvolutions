'''Batch tests'''

import os
import time
import sys

import numpy as np

import sgd_equivariance as se


def main():
	opt = {}
	opt['ortho_loss'] = False
	accs = []
	N_trials = 10
	#for i in xrange(N_trials):
	#	acc = se.main(opt)
	#	accs.append(acc)
	#	np.save('./batch_tests/no_ortho.npy', accs)
	yes = np.load('./batch_tests/no_ortho.npy')
	no = np.load('./batch_tests/yes_ortho.npy')
	
	print np.mean(yes), np.std(yes)
	print np.mean(no), np.std(no)
	

if __name__ == '__main__':
	main()