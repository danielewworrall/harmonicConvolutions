'''Visualizing filters'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

def main():
	k = 20
	phi = np.pi/20.
	G0, A0, G1, A1 = get_complex_basis(k=k)
	X0 = G0*np.cos(A0+phi)
	Y0 = G0*np.sin(A0+phi)
	X1 = G1*np.cos(A1+phi)
	Y1 = G1*np.sin(A1+phi)
	X = X0 + X1
	Y = Y0 + Y1
	#angle = np.pi*145./180.
	#Q2 = Q[...,0]*np.cos(angle) + Q[...,1]*np.sin(angle)
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)/(k/5.)
	plt.figure(1)
	plt.quiver(X,Y)
	#plt.imshow(np.squeeze(Q2), cmap='gray', interpolation='nearest')
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
	G0 = gaussian_derivative(x, y, x)
	A0 = np.arctan2(y, x)# + 0.1*2.*np.pi
	G1 = gaussian_derivative(x, y, y)
	A1 = np.arctan2(y, x)# + np.pi/2.# + 0.1*2.*np.pi
	return G0, A0, G1, A1

def gaussian_derivative(x,y,direction):
    return -2*direction*np.exp(-(x**2 + y**2))

def gaussian(x,y):
    return np.exp(-(x**2 + y**2))

if __name__ == '__main__':
	main()
























