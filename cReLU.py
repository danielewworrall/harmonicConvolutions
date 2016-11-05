'''Complex functions'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

def cReLU(x,y,b):
	r = np.sqrt(x**2 + y**2)
	x_, y_ = x/r, y/r
	return np.maximum(r+b,0)*x_, np.maximum(r+b,0)*y_

n = 20
X = np.zeros((n,n))
Y = np.zeros((n,n))
for i, x in enumerate(np.linspace(-3,3,num=n)):
	for j, y in enumerate(np.linspace(-3,3,num=n)):
		x_, y_ = cReLU(x,y,-0.5)
		X[j,i] = x_
		Y[j,i] = y_

plt.quiver(X,Y)
plt.show()