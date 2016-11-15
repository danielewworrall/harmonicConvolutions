'''Plot ablation'''

import os
import sys
import time

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
sns.set_style('whitegrid')

def run():
	data_dir = './'
	for pend in ['U', 'V', 'rot', 'W']:
		data_Z = np.load(data_dir + 'ablation_' + pend + '.npz')
		n = data_Z['acc'].shape[0]
		plt.plot(12000*np.linspace(1./6,n/6.,n), 100.*(1-data_Z['acc'])) #/np.amax(data_Z['acc']))
	plt.legend(['CNN-20','H-Net-10','H-Net (ours)', 'H-Net-10+DA'], fontsize=20)
	plt.xlabel('Training size', fontsize=20)
	plt.ylabel('Normalized test error', fontsize=20)
	plt.tick_params(axis='both', which='major', labelsize=20)
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	run()