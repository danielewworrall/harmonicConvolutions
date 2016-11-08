'''Make figures for paper'''

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns


def get_complex_rotated_filters(R, psi, filter_size):
    """Return a complex filter of the form $u(r,t,psi) = R(r)e^{im(t-psi)}"""
    filters = {}
    k = filter_size
    for m, r in R.iteritems():
        rsh = r.shape
        print rsh
        # Get the basis matrices
        cmasks, smasks = get_complex_basis_matrices(filter_size, order=m)
        print cmasks.shape
        # Reshape and project taps on to basis
        cosine = np.reshape(cmasks, [k*k, rsh[0]])
        sine = np.reshape(smasks, [k*k, rsh[0]])
        # Project taps on to rotational basis
        print rsh
        r = np.reshape(r, [rsh[0],rsh[1]*rsh[2]])
        ucos = np.reshape(np.dot(cosine, r), [k, k, rsh[1], rsh[2]])
        usin = np.reshape(np.dot(sine, r), [k, k, rsh[1], rsh[2]])
        # Rotate basis matrices
        cosine = np.cos(psi[m])*ucos + np.sin(psi[m])*usin
        sine = -np.sin(psi[m])*ucos + np.cos(psi[m])*usin
        filters[m] = (cosine, sine)
    return filters

def get_complex_basis_matrices(filter_size, order=1):
    """Return complex basis component e^{imt} (ODD sizes only).
    
    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    order: rotation order (default 1)
    """
    k = filter_size
    lin = np.linspace((1.-k)/2., (k-1.)/2., k)
    X, Y = np.meshgrid(lin, lin)
    R = np.sqrt(X**2 + Y**2)
    unique = np.unique(R)
    tap_length = unique.shape[0]
    theta = np.arctan2(-Y, X)
    
    # There will be a cosine and quadrature sine mask
    cmasks = []
    smasks = []
    for i in xrange(tap_length):
        if order == 0:
            # For order == 0 there is nonzero weight on the center pixel
            cmask = (R == unique[i])*1.
            cmasks.append(cmask)
            smask = (R == unique[i])*0.
            smasks.append(smask)
        elif order > 0:
            # For order > 0 re is zero weights on the center pixel
            if unique[i] != 0.:
                cmask = (R == unique[i])*np.cos(order*theta)
                cmasks.append(cmask)
                smask = (R == unique[i])*np.sin(order*theta)
                smasks.append(smask)
    cmasks = np.stack(cmasks, axis=-1)
    cmasks = np.reshape(cmasks, [k,k,tap_length-(order>0)])
    smasks = np.stack(smasks, axis=-1)
    smasks = np.reshape(smasks, [k,k,tap_length-(order>0)])
    return cmasks, smasks


def radial_profile(order=1, n=5):
    """Plot a radial profile of size nxn for an rotation order
    order filter
    """
    #k = (n+1)/2
    #triang = (k*(k+1))/2 - (order>0)
    lin = np.linspace((1.-n)/2., (n-1.)/2., n)
    X, Y = np.meshgrid(lin, lin)
    R = np.sqrt(X**2 + Y**2)
    unique = np.unique(R)
    triang = unique.shape[0] - (order>0)

    R = {order : np.reshape(np.exp(-(unique[(order>0):]**2.)/triang), (triang,1,1))}
    psi = {order : 0.}
    W = get_complex_rotated_filters(R, psi, filter_size=n)

    #plt.figure(1)
    #plt.imshow(W[order][0][:,:,0,0], cmap='gray', interpolation='nearest')
    #plt.axis('off')
    '''
    plt.figure(2)
    plt.imshow(W[order][1][:,:,0,0], cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.figure(3)
    R = np.hstack((1., np.squeeze(R[order])))
    print len(sns.color_palette("muted"))
    '''
    print R
    plt.scatter(unique, R)
    plt.xlim([0,np.amax(unique)])
    plt.ylim([0,1.1])
    # Plot circle
    plt.figure(2)
    t = np.linspace(0,2*np.pi,num=100)
    for u in unique:
        x = (n/2.-0.5)+u*np.cos(t)
        y = (n/2.-0.5)+u*np.sin(t)
        plt.plot(x,y)
    plt.show()

def radial_color(order=1, n=5):
    """Plot a radial profile of size nxn for an rotation order
    order filter
    """
    lin = np.linspace((1.-n)/2., (n-1.)/2., n)
    X, Y = np.meshgrid(lin, lin)
    R = np.sqrt(X**2 + Y**2)
    unique = np.unique(R)
    triang = unique.shape[0] - (order>0)
    theta = np.arctan2(-Y, X)
    
    palette = sns.color_palette("muted")
    weights = np.zeros((5,5,3))
    for i, u in enumerate(unique):
        mask = np.reshape((R == u)*1., (5,5,1))
        pal = np.reshape(np.asarray(palette[i]), (1,1,3))
        weights += mask*pal
    '''
    plt.figure(1)
    plt.imshow(weights[:,:,1], cmap='gray', interpolation='nearest')
    # Plot circle
    palette = sns.color_palette("Paired")
    t = np.linspace(0,2*np.pi,num=100)
    for k, u in enumerate(unique):
        x = (n/2.-0.5)+u*np.cos(t)
        y = (n/2.-0.5)+u*np.sin(t)
        plt.plot(x,y,color=palette[k])
        i, j = np.nonzero((R==u))
        plt.scatter(i, j, color=palette[k], s=60)
        plt.axis('off')
    '''
    '''
    plt.figure(1)
    palette = sns.color_palette("Paired")
    palette = np.reshape(np.asarray(palette), [6,1,3])
    plt.imshow(palette, interpolation='nearest')
    plt.axis('off')
    plt.show()
    '''
    
    for i, u in enumerate(unique):
        plt.figure(i+2)
        sns.set_style("white")
        mask = (R == u)*1.*np.cos(theta)
        #mask += 2*(1-mask)
        plt.imshow(mask, cmap='gray', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.xlim([-0.5,4.5])
        plt.ylim([-0.5,4.5])
    plt.show()
    '''
    masks = np.zeros((5,5,6))
    for i, u in enumerate(unique):
        mask = (R == u)*1.
        masks[:,:,i] = mask*np.cos(theta) + 2*(1-mask)
    mask = np.reshape(masks, [25,6])
    sns.set_style("white")
    plt.imshow(mask.T, cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    #plt.xlim([-0.5,24.5])
    #plt.ylim([-0.5,4.5])
    plt.show()
    '''

if __name__ == '__main__':
    #radial_profile(1,5)
    radial_color()



















