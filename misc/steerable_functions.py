'''Steerable functions'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt


def main():
	N = 51
	t = np.pi/4.
	alpha = -2./2.
	bw2 = (N)**2
	# Create meshgrid
	lin = np.linspace(-N,N,2*N+1)
	X, Y = np.meshgrid(lin, lin)
	Y = -Y
	theta = np.arctan2(Y, X)
	R2 = X**2 + Y**2
	# Get basis vectors
	G = np.exp(-R2/bw2)
	R = np.cos(alpha*np.log(np.maximum(R2,1e-12))) 
	
	plt.figure(1)
	plt.imshow(G, interpolation='nearest', cmap='gray')
	plt.figure(2)
	plt.imshow(R, interpolation='nearest', cmap='gray')
	plt.show()


if __name__ == '__main__':
	main()