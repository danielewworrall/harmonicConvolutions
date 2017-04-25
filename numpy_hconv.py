"""Numpy h-conv"""

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt


def visualize_weights():
   """View the random weights generated from the prior"""
   ksize = 3
   n_channels = 1
   max_order = 1
   stddev = 0.4
   n_rings = None
   phase = True
   name = 'test'

   x = np.random.randn(4,3,3,1,1,1)
   xsh = x.shape
   shape = [ksize, ksize, xsh[5], n_channels]
   Q = get_weights_dict(shape, max_order, std_mult=stddev, n_rings=n_rings, name='W'+name)
   if phase == True:
       P = get_phase_dict(xsh[5], n_channels, max_order, name='phase'+name)
   else:
       P = None
   W = get_filters(Q, filter_size=ksize, P=P, n_rings=n_rings)

   wr = np.squeeze(W[1][0])
   wi = np.squeeze(W[1][1]).T[::-1,:]
   print wr - wi
   print


def get_weights_dict(shape, max_order, std_mult=0.4, n_rings=None, name='W'):
   """Return a dict of weights.

   shape: list of filter shape [h,w,i,o] --- note we use h=w
   max_order: returns weights for m=0,1,...,max_order, or if max_order is a
   tuple, then it returns orders in the range.
   std_mult: He init scaled by std_mult (default 0.4)
   name: (default 'W')
   dev: (default /cpu:0)
   """
   if isinstance(max_order, int):
      orders = xrange(-max_order, max_order+1)
   else:
      diff = max_order[1]-max_order[0]
      orders = xrange(-diff, diff+1)
   weights_dict = {}
   for i in orders:
      if n_rings is None:
         n_rings = np.maximum(shape[0]/2, 2)
      sh = [n_rings,] + shape[2:]
      nm = name + '_' + str(i)
      weights_dict[i] = get_weights(sh, std_mult=std_mult, name=nm)
   return weights_dict


def get_phase_dict(n_in, n_out, max_order, name='b'):
   """Return a dict of phase offsets"""
   if isinstance(max_order, int):
       orders = xrange(-max_order, max_order+1)
   else:
       diff = max_order[1]-max_order[0]
       orders = xrange(-diff, diff+1)
   phase_dict = {}
   for i in orders:
       init = np.random.rand(1,1,n_in,n_out) * 2. *np.pi
       phase = init*np.ones((1,1,n_in,n_out))
       phase_dict[i] = phase
   return phase_dict



def n_samples(filter_size):
   return np.maximum(np.ceil(np.pi*filter_size), 101) # <-------------------------------------One source of instability


def L2_grid(center, shape):
   # Get neighbourhoods
   lin = np.arange(shape)+0.5
   J, I = np.meshgrid(lin, lin)
   I = I - center[1]
   J = J - center[0]
   return np.vstack((np.reshape(I, -1), np.reshape(J, -1)))


def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W'):
   """Initialize weights variable with He method

   filter_shape: list of filter dimensions
   W_init: numpy initial values (default None)
   std_mult: multiplier for weight standard deviation (default 0.4)
   name: (default W)
   device: (default /cpu:0)
   """
   if W_init == None:
      stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:3]))
      W = stddev*np.random.randn(*filter_shape)
   return W


def get_interpolation_weights(filter_size, m, n_rings=None):
   """Resample the patches on rings using Gaussian interpolation"""
   if n_rings is None:
      n_rings = np.maximum(filter_size/2, 2)
   radii = np.linspace(m!=0, n_rings-0.5, n_rings) # <-------------------------Look into m and n_rings-0.5
   # We define pixel centers to be at positions 0.5
   foveal_center = np.asarray([filter_size, filter_size])/2.
   # The angles to sample
   N = n_samples(filter_size)
   lin = (2*np.pi*np.arange(N))/N
   # Sample equi-angularly along each ring
   ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])
   # Create interpolation coefficient coordinates
   coords = L2_grid(foveal_center, filter_size)
   # Sample positions wrt patch center IJ-coords
   radii = radii[:,np.newaxis,np.newaxis,np.newaxis]
   ring_locations = ring_locations[np.newaxis,:,:,np.newaxis]
   diff = radii*ring_locations - coords[np.newaxis,:,np.newaxis,:]
   dist2 = np.sum(diff**2, axis=1)
   # Convert distances to weightings
   bandwidth = 0.5
   weights = np.exp(-0.5*dist2/(bandwidth**2))
   # Normalize
   return weights/np.sum(weights, axis=2, keepdims=True)


def get_filters(R, filter_size, P=None, n_rings=None):
   """Perform single-frequency DFT on each ring of a polar-resampled patch"""
   k = filter_size
   filters = {}
   N = n_samples(k)
   from scipy.linalg import dft
   for m, r in R.iteritems():
      rsh = r.shape
      # Get the basis matrices
      weights = get_interpolation_weights(k, m, n_rings=n_rings)
      DFT = dft(N)[m,:]
      LPF = np.dot(DFT, weights).T
      cosine = np.real(LPF).astype(np.float32)
      sine = np.imag(LPF).astype(np.float32)
       # Project taps on to rotational basis
      r = np.reshape(r, np.stack([rsh[0],rsh[1]*rsh[2]]))
      ucos = np.reshape(np.dot(cosine, r), np.stack([k, k, rsh[1], rsh[2]]))
      usin = np.reshape(np.dot(sine, r), np.stack([k, k, rsh[1], rsh[2]]))
      if P is not None:
         # Rotate basis matrices
         ucos_ = np.cos(P[m])*ucos + np.sin(P[m])*usin
         usin = -np.sin(P[m])*ucos + np.cos(P[m])*usin
         ucos = ucos_
      filters[m] = (ucos, usin)
   return filters


if __name__ == '__main__':
   visualize_weights()
