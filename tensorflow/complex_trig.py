'''Complex plane trigonometry'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

theta = np.linspace(0., 2.*np.pi, num=100)
c = np.cos(theta)[:,np.newaxis]
s = np.sin(theta)[:,np.newaxis]

a = np.asarray([3,-1])[np.newaxis,:]
b = np.asarray([0,2])[np.newaxis,:]

y = c*a + s*b
d = (a+b)

plt.figure(1)
plt.scatter(y[:,0], y[:,1], color='b')
plt.scatter(a[:,0], a[:,1], color='r')
plt.scatter(b[:,0], b[:,1], color='y')
plt.scatter(d[:,0], d[:,1], color='g')
plt.figure(2)
plt.plot(theta, y[:,0])
plt.show()