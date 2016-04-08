'''Permutation groups'''

import os
import sys
import time

import cv2
import numpy
import scipy.linalg as scilin
import scipy.signal as scisig
import skimage.io as skio

from matplotlib import pyplot as plt

class RotConv():
	def __init__(self):
		pass
	
	def rotate(self, angle):
		R = numpy.asarray([[numpy.cos(angle), -numpy.sin(angle)],[numpy.sin(angle), numpy.cos(angle)]])
		return R

	def correlate(self, im, kernel):	
		res = []
		for i in xrange(3):
			res.append(scisig.correlate2d(im[...,i], kernel))
		return numpy.sum(numpy.stack(res, 2), axis=2)
	
	def rotation(self, angle, n):
		rot = numpy.eye(n)
		R = self.rotate(angle)
		for i in xrange(n/2):
			rot[i*2:(i+1)*2,i*2:(i+1)*2] = R
		return rot
	
	def reform(self, M):
		'''Reshape the matrix to return a 4-tensor with new subvectors for angles'''
		mSh = M.shape
		mSh = mSh + (2,)
		mSh = numpy.asarray(mSh)
		mSh[1] -= 1
		res = numpy.zeros(mSh)
		res[...,0] = M[:,:-1,:]
		res[...,1] = M[:,1:,:]
	
	def unravel(self, M):
		P = numpy.zeros((9,9))
		P[0,0] = 1
		P[1,1] = 1
		P[2,2] = 1
		P[3,5] = 1
		P[4,8] = 1
		P[5,7] = 1
		P[6,6] = 1
		P[7,3] = 1
		P[8,4] = 1
		M = M.reshape((9,1))
		return numpy.dot(P, M)
	
	def ravel(self, M):
		P = numpy.zeros((9,9))
		P[0,0] = 1
		P[1,1] = 1
		P[2,2] = 1
		P[3,5] = 1
		P[4,8] = 1
		P[5,7] = 1
		P[6,6] = 1
		P[7,3] = 1
		P[8,4] = 1
		M = numpy.dot(P.T, M)
		return M.reshape((3,3))
	
	def permuteFourier(self, F):
		P = numpy.zeros((9,9))
		P[0,0] = 1
		P[1,4] = 1
		P[2,1] = 1
		P[3,7] = 1
		P[4,2] = 1
		P[5,6] = 1
		P[6,3] = 1
		P[7,5] = 1
		P[8,8] = 1
		return numpy.dot(P, F)

	def getQ(self, n):
		Q = numpy.eye(n, dtype=numpy.complex)
		Q[:n-1,:n-1] = scilin.dft(n-1)/(numpy.sqrt(n-1.))
		P = self.permuteFourier(Q)
		u = numpy.asarray([[1,1],[1j,-1j]])
		U = numpy.eye(n, dtype=numpy.complex)
		U[2:4,2:4] = u
		U[4:6,4:6] = u
		U[6:8,6:8] = u
		Q = numpy.real(numpy.dot(U,P))
		return Q
	
	def getRotation(self, angle):
		R = numpy.eye(9)
		t = angle/8.
		R[1,1] = numpy.cos(4*t)
		R[2:4,2:4] = self.rotate(t)
		R[4:6,4:6] = self.rotate(2*t)
		R[6:8,6:8] = self.rotate(3*t)
		return R
	
	def transformQ(self, Q, x, y):
		m = numpy.dot(Q,x)
		w = numpy.dot(Q,y)
		return (m, w)
	
	def decomposeVectors(self, m):
		'''Decompose vector into angles and magnitudes'''
		vecs = numpy.array_split(m, 5)
		Vecs = numpy.hstack((vecs[0], vecs[1], vecs[2], vecs[3]))
		mags = numpy.sqrt(numpy.sum(Vecs**2, axis=0))
		Vecs = Vecs/(mags + 1e-5)
		return (Vecs, mags, vecs[4][0,0])
	
	def genData(self):
		# Generate the data
		kernel = cv2.getGaborKernel((3,3), 5., numpy.pi/4., 10., 0.1)
		kernelUnrav = self.unravel(kernel)
		patch = numpy.random.randn(3,3)
		patchUnrav = self.unravel(patch)
		return (kernel, kernelUnrav, patch, patchUnrav)
	
	def lazyRotate(self, Q, kernelUnrav, patch, N):
		'''Multiple the patch with a rotated filter, the brute force way'''
		y = []
		for i in xrange(N):
			M = numpy.dot(Q, kernelUnrav)
			shift = self.getRotation(0.2*2*numpy.pi*i)
			recon = numpy.dot(Q.T, numpy.dot(shift, numpy.dot(Q, kernelUnrav)))
			reconr = numpy.real(self.ravel(recon))
			y.append(numpy.sum(reconr*patch))
		return y
		
	def demo1(self):
		'''Using a simple oriented Gabor filter'''
		# Generate demo data
		im = skio.imread('/home/daniel/Code/LieGroups/kingfisher.jpg')
		kernel = cv2.getGaborKernel((3,3), 5., numpy.pi/4., 10., 0.1)
		kernelUnrav = rc.unravel(kernel)
		kernel, kernelUnrav, __, __ = self.genData()
		# Generate transformation matrix
		Q = self.getQ(9)
		# Plotting
		fig = plt.figure(1)
		plt.ion()
		plt.show()
		# Loop through each rotation of the filter
		for i in xrange(100):
			# Projection of filter onto rotation basis
			M = numpy.dot(Q, kernelUnrav)
			shift = self.getRotation(0.2*2*numpy.pi*i)
			recon = numpy.dot(Q.T, numpy.dot(shift, numpy.dot(Q, kernelUnrav)))
			reconr = numpy.real(self.ravel(recon))
			# Perform the correlation/convolution
			res = self.correlate(im, reconr)
			plt.imshow(res, cmap='gray', interpolation='nearest')
			plt.draw()
			raw_input(i)
	
	def demo2(self):
		'''The slow patch mixture of cosine demo'''
		N = 100
		# Generate the data
		kernel = cv2.getGaborKernel((3,3), 5., numpy.pi/4., 10., 0.1)
		kernelUnrav = self.unravel(kernel)
		patch = numpy.random.randn(3,3)
		kernel, kernelUnrav, patch, __ = self.genData()
		# Generate transformation matrix
		Q = self.getQ(9)
		# Generate response
		y = self.lazyRotate(Q, kernelUnrav, patch, N)
		# Plot
		fig = plt.figure(1)
		plt.plot(numpy.arange(1000), y)
		plt.show()
	
	def demo3(self):
		'''The fast patch mixture of cosines demo'''
		N = 100
		# Generate the data
		kernel, kernelUnrav, patch, patchUnrav = self.genData()
		# Generate transformation on rotation basis
		Q = rc.getQ(9)
		s = time.time()
		Qm, Qpatch = self.transformQ(Q, kernelUnrav, patchUnrav)
		# Decompose signal into magnitude and normalized vectors
		mVec, mMag, mConst = self.decomposeVectors(Qm)
		patchVec, patchMag, patchConst = self.decomposeVectors(Qpatch)
		mag = mMag*patchMag
		# Get angle
		angles = numpy.arccos(numpy.sum(patchVec*mVec, axis=0))
		# Get signs
		rot90 = numpy.asarray([[0,-1],[1,0]])
		sign = (numpy.sum(numpy.dot(rot90, patchVec)*mVec, axis=0) <= 0)*2. - 1.
		angles = angles*sign
		y = numpy.zeros((N,))
		artefact0 = Qm[0]*Qpatch[0]
		mag[0] = Qm[1]*Qpatch[1]
		sign[0] = 1
		phiBase = 0.2*2.*numpy.pi*numpy.asarray([4,1,2,3])/8.
		for i in xrange(N):
			phi = phiBase*i
			y[i] = numpy.dot(mag, numpy.cos(phi - angles)) + artefact0
		y += mConst*patchConst
		print(time.time()-s)
		s = time.time()
		z = self.lazyRotate(Q, kernelUnrav, patch, N)
		print(time.time() - s)
		fig = plt.figure(1)
		plt.plot(numpy.arange(N), y)
		plt.plot(numpy.arange(N), z, '--')
		plt.show()

if __name__ == '__main__':
	rc = RotConv()
	#rc.demo1()
	#rc.demo2()
	rc.demo3()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	
	