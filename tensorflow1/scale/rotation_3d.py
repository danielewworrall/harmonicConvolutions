'''3D rotation matrices'''

import os
import sys
import time

import numpy as np

def transmat(phi, theta, psi, shiftmat=None):
    """ Compute the 3D transformation matrix with rotations about x,y and z axii
        phi = x-axis-rot, 
        theta = y-axis-rot, 
        psi = z-axis-rot
    
    """
    batch_size = phi.shape[0]
    assert batch_size==theta.shape[0] and batch_size==psi.shape[0], 'must have same number of angles for x,y and z axii'
    assert phi.ndim==1 and theta.ndim==1 and psi.ndim==1, 'must be 1 dimensional array'

    if shiftmat is None:
        shiftmat = np.zeros([batch_size,3,1])

    rotmat = np.zeros([batch_size, 3,3])
    rotmat[:,0,0] = np.cos(theta)*np.cos(psi)
    rotmat[:,0,1] = np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi)
    rotmat[:,0,2] = np.sin(phi)*np.sin(psi) - np.cos(phi)*np.sin(theta)*np.cos(psi)
    rotmat[:,1,0] = -np.cos(theta)*np.sin(psi)
    rotmat[:,1,1] = np.cos(phi)*np.cos(psi) - np.sin(phi)*np.sin(theta)*np.sin(psi)
    rotmat[:,1,2] = np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)
    rotmat[:,2,0] = np.sin(theta)
    rotmat[:,2,1] = -np.sin(phi)*np.cos(theta)
    rotmat[:,2,2] = np.cos(phi)*np.cos(theta)

    transmat = np.concatenate([rotmat, shiftmat],2)
    return np.reshape(transmat, [batch_size, -1]).astype(np.float32)


def transmat_a2b(transmat_src, transmat_trg):
    """ Compute the 3D transformation matrix 
        that inverses SRC transformation and applies TRG transformation

    """
    transmat_src = np.reshape(transmat_src, [3,4])
    transmat_trg = np.reshape(transmat_trg, [3,4])
    R_src = transmat_src[:,0:3]
    t_src = transmat_src[:,3:4]
    R_trg = transmat_trg[:,0:3]
    t_trg = transmat_trg[:,3:4]
    R_res = np.dot(R_src.T, R_trg)
    t_res = -t_src + t_trg # TODO this is probably not correct. assume no translation at the moment
    result = np.concatenate([R_res, t_res], 1)
    return result.flatten()


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
