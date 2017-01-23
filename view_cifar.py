'''Visualize cifar10'''

import os
import sys
import time

import numpy as np
np.set_printoptions(threshold=np.nan)

from matplotlib import pyplot as plt


def main():
    folder = './cifar10/cifar_numpy/'
    folderp = './cifar10/cifar_preproc/'
    fnames = ['trainX.npy','validX.npy','testX.npy']
    train = np.load(folder+fnames[0])
    valid = np.load(folder+fnames[1])
    test = np.load(folder+fnames[2])
    print train.shape
    print valid.shape
    print test.shape
    data = np.vstack([train, valid])
    # Global contrast normalization parameters
    mean, stddev = GCN(data)
    data = (data-mean)/(stddev + 1e-6)
    mean, stddev = GCN(test)
    test = (test-mean)/(stddev + 1e-6)
    # Zero-phase PCA
    U, s = get_ZCA_params(data)
    data = ZCA(data, U, s)
    test = ZCA(test, U, s)
    train = data[:44500,...]
    valid = data[44500:,...]
    print train.shape
    print valid.shape
    print test.shape
    np.save(folderp+fnames[0], train)
    np.save(folderp+fnames[1], valid)
    np.save(folderp+fnames[2], test)
    
    
def GCN(x, eps=1e-6):
	 """Global contrast normalization"""
	 mean = np.mean(x, 1, keepdims=True)
	 stddev = np.std(x, 1, keepdims=True)
	 stddev[stddev < eps] = 1.
	 return mean, stddev


def get_ZCA_params(x):
    cov = np.dot(x.T,x)/np.shape(x)[0]
    [U,s,V] = np.linalg.svd(cov)
    return U, s


def ZCA(x, U, s, eps=1e-6):
    """Zero-phase PCA"""
    x_rot = np.dot(U.T, x.T)
    x_PCA = np.dot(np.diag(1/np.sqrt(s + eps)),x_rot)
    x_ZCA = np.dot(U,x_PCA)
    return x_ZCA.T


folder = './cifar10/cifar_numpy'
data = np.load(folder + '/trainX.npy')

fig = plt.figure(1)
idx = np.random.randint(44500)
datum = data[idx,:]
plt.imshow(np.reshape(datum, (32,32,3)))
plt.show()

mean, stddev = GCN(datum[np.newaxis,:])
datum = (datum - mean) / stddev

fig = plt.figure(2)
plt.imshow(np.reshape(datum, (32,32,3))[:,:,1])
plt.show()

data = (data - mean) / stddev
U, s = get_ZCA_params(data)

fig = plt.figure(3)
dataZ = ZCA(data, U, s, eps=1e-6)
datum = dataZ[idx,:]
plt.imshow(np.reshape(datum, (32,32,3))[:,:,0], cmap='gray')

fig = plt.figure(4)
dataZ = ZCA(data, U, s, eps=1e-3)
datum = dataZ[idx,:]
plt.imshow(np.reshape(datum, (32,32,3))[:,:,0], cmap='gray')


fig = plt.figure(5)
dataZ = ZCA(data, U, s, eps=1e-1)
datum = dataZ[idx,:]
plt.imshow(np.reshape(datum, (32,32,3))[:,:,0], cmap='gray')
plt.show()















