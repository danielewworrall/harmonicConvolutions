'''3D rotation matrices'''

import os
import sys
import time

import numpy as np


def rot3d(phi, theta):
	"""Compute the 3D rotation matrix for a roll-less transformation"""
	rotY = np.asarray([[np.cos(phi),0.,-np.sin(phi)],
							[0.,1.,0.],
							[np.sin(phi),0.,np.cos(phi)]])
	rotZ = np.asarray([[np.cos(theta),np.sin(theta),0.],
							[-np.sin(theta),np.cos(theta),0.],
							[0.,0.,1]])
	return np.dot(rotZ, rotY)


def rot3d_a2b(phi1, theta1, phi2, theta2):
	"""Compute the 3D rotation matrix for a roll-less transformation from A to B"""
	rot1_inv = rot3d(phi1, theta1).T
	rot2 = rot3d(phi2, theta2)
	return np.dot(rot2, rot1_inv)


def test():
	# Generate random angles
	theta1 = 2.*np.pi*np.random.rand()
	phi1 = np.pi*np.random.rand()
	
	theta2 = 2.*np.pi*np.random.rand()
	phi2 = np.pi*np.random.rand()
	
	r1 = rot3d_a2b(phi1, theta1, phi2, theta2)
	r2 = rot3d_a2b(phi2, theta2, phi1, theta1)
	print r1
	print r1 - r2.T


if __name__ == '__main__':
	test()