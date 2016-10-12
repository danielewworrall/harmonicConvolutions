'''Visualizing filters'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

def main():
	k = 20.
	phi = 0.*np.pi
	G, A = get_complex_basis(k=k)

	X = np.cos(A-phi)
	Y = np.sin(A-phi)
	
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)/(k/5.)
	plt.figure(1)
	plt.quiver(X,Y)
	plt.imshow(G)
	
	plt.show()

def get_basis(k=3,n=2):
    """Return a tensor of steerable filter bases"""
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)/(k/5.)
    x, y = np.meshgrid(lin, lin)
    gdx = gaussian_derivative(x, y, x)
    gdy = gaussian_derivative(x, y, y)
    G0 = np.reshape(gdx/np.sqrt(np.sum(gdx**2)), [k,k,1])
    G1 = np.reshape(gdy/np.sqrt(np.sum(gdy**2)), [k,k,1])
    return np.concatenate([G0,G1], axis=2)

def get_complex_basis(k=3,n=2):
	"""Return a tensor of steerable filter bases"""
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)/(k/5.)
	x, y = np.meshgrid(lin, lin)
	G = gaussian_derivative(x, y, x)
	A = np.arctan2(y, x)
	return G, A

def gaussian_derivative(x,y,direction):
	r2 = x**2 + y**2
	return -2*np.sqrt(r2)*np.exp(-r2)

def gaussian(x,y):
    return np.exp(-(x**2 + y**2))

if __name__ == '__main__':
	main()
























