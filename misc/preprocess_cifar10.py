'''Preprocess CIFAR10 with global contrast normalization and ZCA'''

import os
import sys

import numpy as np

from matplotlib import pyplot as plt

def main():
    folder = '../cifar10/cifar_numpy/'
    folderp = '../cifar10/cifar_preproc/'
    fnames = ['trainX.npy','validX.npy','testX.npy']
    train = np.load(folder+fnames[0])
    valid = np.load(folder+fnames[1])
    test = np.load(folder+fnames[2])

    data = np.vstack([train, valid])
    # Global contrast normalization parameters---recalculate stats for both sets
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

    np.save(folderp+fnames[0], train)
    np.save(folderp+fnames[1], valid)
    np.save(folderp+fnames[2], test)
    
    
def GCN(x):
    """Global contrast normalization"""
    mean = np.mean(x, 1, keepdims=True)
    stddev = np.std(x, 1, keepdims=True)
    return mean, stddev


def get_ZCA_params(x):
    cov = np.dot(x.T,x)/np.shape(x)[0]
    [U,s,V] = np.linalg.svd(cov)
    return U, s


def ZCA(x, U, s):
    """Zero-phase PCA"""
    x_rot = np.dot(U.T, x.T)
    x_PCA = np.dot(np.diag(1/np.sqrt(s + 1e-6)),x_rot)
    x_ZCA = np.dot(U,x_PCA)
    return x_ZCA.T
    


if __name__ == '__main__':
    main()
