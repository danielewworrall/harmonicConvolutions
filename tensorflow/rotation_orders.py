'''Rotation order maths---figure out how rotation orders combine'''

import os
import sys
import time

import numpy as np
import scipy.signal as scisig
import skimage.io as skio
import skimage.color as skco
import scipy.ndimage.interpolation as sciint

from matplotlib import pyplot as plt

def steerable_filter(k, order):
	lin = np.linspace((1.-k)/2., (k-1.)/2., k)
	X, Y = np.meshgrid(lin, lin)
	R = np.sqrt(X**2 + Y**2)
	unique = np.unique(R)
	theta = np.arctan2(-Y, X)
	
	cos_filter = np.cos(order*theta)
	sin_filter = np.sin(order*theta)
	if order != 0:
		c = k/2
		cos_filter[c,c] = 0.
		sin_filter[c,c] = 0.
	cos_filter = cos_filter/np.sum(cos_filter**2.)
	sin_filter = sin_filter/np.sum(sin_filter**2.)
	return cos_filter, sin_filter


im = skio.imread('../LieNets/kingfisher.jpg')
im = skco.rgb2gray(im)[100:200,500:600]

for angle in [0,45,90,135,180]:
	kernels = steerable_filter(3, 1)
	im_ = sciint.rotate(im, angle, reshape=False)
	imc = scisig.fftconvolve(kernels[0], im_, mode='valid')
	ims = scisig.fftconvolve(kernels[1], im_, mode='valid')
	print np.std(imc), np.std(ims)
	
	
	kernels = steerable_filter(3, 1)
	imrr = scisig.fftconvolve(kernels[0], imc, mode='valid')
	imii = scisig.fftconvolve(kernels[1], ims, mode='valid')
	imri = scisig.fftconvolve(kernels[0], ims, mode='valid')
	imir = scisig.fftconvolve(kernels[1], imc, mode='valid')
	imc = imrr - imii
	ims = imir + imri
	
	
	kernels = steerable_filter(3, -2)
	imrr = scisig.fftconvolve(kernels[0], imc, mode='valid')
	imii = scisig.fftconvolve(kernels[1], ims, mode='valid')
	imri = scisig.fftconvolve(kernels[0], ims, mode='valid')
	imir = scisig.fftconvolve(kernels[1], imc, mode='valid')
	imc = imrr - imii
	ims = imir + imri
	
	print np.std(imc), np.std(ims)
	im_mod = np.sqrt(imc**2 + ims**2)
	plt.figure(angle)
	plt.imshow(im_mod, cmap='gray', interpolation='nearest')
	
plt.show()