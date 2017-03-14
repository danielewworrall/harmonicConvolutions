'''3D rotation matrices'''

import os
import sys
import time

import numpy as np


def transmat(xyzrot, xyzfactor=None, shiftmat=None):
    """ Compute the 3D transformation matrix with 
        rotations about x,y and z axii and scales about x, y and z axii.

        xyzrot = x,y,z-axis rotation
        xyzfactor = x,y,z-axis scale
        
    """
    batch_size = xyzrot.shape[0]

    rotmat = get_3drotmat(xyzrot)

    if xyzfactor is None:
        xyzfactor = np.ones([batch_size, 3])

    assert xyzfactor.shape[0]==batch_size, 'xyzfactor must have scale factor for each datapoint in the minibatch'

    if shiftmat is None:
        shiftmat = np.zeros([batch_size,3,1])

    scalemat = get_3dscalemat(xyzfactor)
    rotscalemat = np.matmul(rotmat, scalemat)

    assert rotscalemat.ndim==3 and rotscalemat.shape[0]==batch_size and rotscalemat.shape[1]==3 and rotscalemat.shape[2]==3, 'matmul works'

    transmat = np.concatenate([rotscalemat, shiftmat],2)
    return np.reshape(transmat, [batch_size, -1]).astype(np.float32)


def transmat_a2b(transmat_src, transmat_trg):
    #TODO
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
