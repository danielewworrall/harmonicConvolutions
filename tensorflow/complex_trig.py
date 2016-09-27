'''Complex plane trigonometry'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

def get_basis_taps(k=3):##################THIS IS ALL WRONG
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
		Wx += mask*tap_x[i]*X
		Wy += mask*tap_y[i]*Y
	
	return (Wx, Wy)

k = 10
s = 0.1

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

plt.figure(1)
plt.imshow(T1, interpolation='nearest')
plt.quiver(T1_x/T1, T1_y/T1)
plt.figure(2)
plt.imshow(T2, interpolation='nearest')
plt.quiver(T2_x/T2, T2_y/T2)
plt.figure(3)
plt.imshow(T, interpolation='nearest')
plt.quiver(T_x/T, T_y/T)
plt.show()