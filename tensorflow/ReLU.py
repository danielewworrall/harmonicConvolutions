'''Augmented ReLU'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

def ReLU(z,b):
    """Augmented ReLU for the complex plane"""
    x, y = z
    r2 = x**2 + y**2
    s = 0.5*(np.sign(r2 - b**2) + 1.)
    x = s*np.sign(x)*(np.abs(x)-b*np.sqrt(x**2+y**2))
    y = s*np.sign(y)*(np.abs(y)-b*np.sqrt(x**2+y**2))
    return x, y

if __name__ == '__main__':
    x = np.linspace(-10,10, num=250)
    X, Y = np.meshgrid(x, x)
    Y = -Y
    
    Z = ReLU((X,Y), 5.)
    R = np.sqrt(Z[0]**2 + Z[1]**2)
    
    plt.figure(1)
    plt.imshow(R, interpolation='nearest')
    plt.show()