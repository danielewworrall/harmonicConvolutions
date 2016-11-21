'''Draw a gaussian derivative filter'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt


def run():
	dim = 51
	m = 1
	theta = 0.
	Greal, Gimag = get_filter(dim, m, theta)
	plt.imshow(Greal, interpolation='nearest', cmap='gray')
	plt.show()

def get_filter(dim, m, theta):
	"""Dim is the side length, m the rotn order and theta the orientation"""
	lin = np.linspace(-(dim/2), (dim/2), num=dim) / (dim/5.)
	x, y = np.meshgrid(lin, lin)
	phi = np.arctan2(y, x)
	r = np.sqrt(x**2 + y**2)
	
	real = np.exp(-r**2)*np.cos(m*phi + theta)
	imag = np.exp(-r**2)*np.sin(m*phi + theta)
	return (real, imag)


if __name__ == '__main__':
	run()
