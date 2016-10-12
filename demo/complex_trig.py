'''Complex plane trigonometry'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

def get_basis_taps(k=3):
	tap_x = np.random.randn(int(0.5*(k**2-1)))
	tap_y = np.random.randn(int(0.5*(k**2-1)))
	
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)
	X, Y = np.meshgrid(lin, lin)
	Y = -Y
	R = np.sqrt(X**2 + Y**2)
	unique = np.unique(R)
	
	Wx = np.zeros((k,k))
	Wy = np.zeros((k,k))
	for i in xrange(len(unique)):
		mask = (R == unique[i])
		Wx += mask*tap_x[i]*X/np.maximum(R,1.)
		Wy += mask*tap_y[i]*Y/np.maximum(R,1.)
	
	return (Wx, Wy)

def double():
	k = 3
	N = 36
	s = 1.
	
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)
	X, Y = np.meshgrid(lin, lin)
	Y = -Y
	
	a = get_basis_taps(k=k)
	b = get_basis_taps(k=k)
	
	plt.figure(1)
	plt.ion()
	plt.show()
	for i in xrange(N):
		plt.cla()
		theta = i*2.*np.pi/N
		Tx = np.cos(theta)*a[0] + np.sin(theta)*a[1]
		Ty = np.cos(theta)*b[0] + np.sin(theta)*b[1]
		T = np.sqrt(Tx**2 + Ty**2)
		plt.imshow(T, cmap='gray', interpolation='nearest')
		plt.quiver(Tx/T, Ty/T)
		plt.draw()
		raw_input(theta)
	
def complex_():
	k = 3
	N = 36
	s = 1.
	
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)
	X, Y = np.meshgrid(lin, lin)
	Y = -Y
	
	R = np.sqrt(X**2 + Y**2)
	
	a = get_basis_taps(k=k)
	b = get_basis_taps(k=k)
	
	T1_x = a[0] - a[1]
	T1_y = a[0] + a[1]
	T1 = np.sqrt(T1_x**2 + T1_y**2)
	
	T2_x = b[0] + b[1]
	T2_y = -b[0] + b[1]
	T2 = np.sqrt(T2_x**2 + T2_y**2)
	
	T_x = T1_x + s*T2_x
	T_y = T1_y + s*T2_y
	T = np.sqrt(T_x**2 + T_y**2)
	
	T_xx = T_x
	T_xy = np.flipud(T_x.T)
	T_yx = np.flipud(T_y.T)
	T_yy = T_y
	
	plt.figure(1)
	plt.ion()
	plt.show()
	for i in xrange(N):
		plt.cla()
		theta = i*2.*np.pi/N
		T_xr = np.cos(theta)*T_xx + np.sin(theta)*T_xy
		T_yr = np.cos(theta)*T_yx + np.sin(theta)*T_yy
		T = np.sqrt(T_xr**2 + T_yr**2)
		plt.imshow(T, cmap='gray', interpolation='nearest')
		plt.quiver(T_xr/T, T_yr/T)
		plt.draw()
		raw_input(theta)


if __name__ == '__main__':
	complex_()