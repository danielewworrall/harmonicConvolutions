'''Filter tests'''

import os
import sys
import time

import cv2
import numpy as np
import scipy as sp
import skimage.color as skco
import skimage.io as skio

from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

import seaborn as sns
colors = ["windows blue", "amber", "greyish", "faded green"]
colors = sns.xkcd_palette(colors)

def view_filters():
	N_range = 2*np.arange(7) + 3
	plt.ion()
	plt.show()
	for N in N_range:
		print N
		Psi = np.load('./filters/rs_'+str(N)+'.npy')
		Psi = np.reshape(Psi, (16,-1))
		plt.imshow(np.dot(Psi,Psi.T), cmap='gray')
		plt.draw()
		raw_input()
		

def magnitude_response():
	image = skio.imread('../images/balloons.jpg')[50:250,150:350]
	image = skco.rgb2gray(image)
	imsh = image.shape
	Theta = np.linspace(0, 2*np.pi, num=360, endpoint=False)
	
	plt.ion()
	plt.show()
	max_error = []
	N_range = 2*np.arange(7) + 3
	for N in N_range:
		print N
		Psi = np.load('./filters/rs_'+str(N)+'.npy')
		#for j in xrange(8):
		for color_num, j in enumerate([0,2,8,10]):
			Y = []
			for theta in Theta:
				#M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),180.*theta/(np.pi),1)
				#image_ = cv2.warpAffine(image,M,image.shape,flags=cv2.INTER_CUBIC)
				scale = np.power(1.132,theta)
				new_shape = (int(imsh[0]),int(imsh[1]*scale))
				image_ = cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
				#margin = ((new_shape[0] - imsh[0])/2, (new_shape[1] - imsh[1])/2)
				#image_ = image_[margin[0]:margin[0]+imsh[0], margin[1]:margin[1]+imsh[1]]
				
				#r1 = fftconvolve(image_, Psi[2*j,...], mode='same')
				#r2 = fftconvolve(image_, Psi[2*j+1,...], mode='same')
				#inv = np.sqrt(r1**2 + r2**22)
				inv = 0.
				for k in [0,1,4,5]:
					inv += fftconvolve(image_, Psi[j+k,...], mode='same')**2
				inv = np.sqrt(inv)
				
				#M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),-180.*theta/(np.pi),1)
				#inv = cv2.warpAffine(inv,M,image.shape,flags=cv2.INTER_CUBIC)
				inv = cv2.resize(inv, imsh, interpolation=cv2.INTER_CUBIC)
				
				Y.append(inv)
			
			ref = Y[0][50:150,50:150]
			errors = []
			phase = []
			for y in Y:
				crop = y[50:150,50:150]
				error = np.mean((ref-crop)**2) / (np.sqrt(np.mean(ref**2))*np.sqrt(np.mean(crop**2)))
				errors.append(error)
			errors = np.asarray(errors)
			#plt.plot(np.power(1.132,Theta), errors, color=colors[color_num])
		#plt.draw()
		#raw_input()
		#plt.clf()
		max_error.append(np.amax(errors))
	plt.semilogy(N_range, max_error)
	plt.xlabel('Filter size')
	plt.ylabel('Maximum error')
	plt.draw()
	raw_input()

def phase_response():
	image = skio.imread('../images/balloons.jpg')[50:250,150:350]
	image = skco.rgb2gray(image)
	Theta = np.linspace(0, 2*np.pi, num=360, endpoint=False)
	
	#plt.ion()
	#plt.show()
	max_error = []
	N_range = 2*np.arange(7) + 3
	N_range = N_range[1:]
	for N in N_range:
		print N
		Psi = np.load('./filters/rs_'+str(N)+'.npy')
		deviations = []
		plt.clf()
		for j in xrange(8):
			phase = []
			for theta in Theta:
				M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),180.*theta/(np.pi),1)
				image_ = cv2.warpAffine(image,M,image.shape,flags=cv2.INTER_CUBIC)
				r1 = fftconvolve(image_, Psi[2*j,...], mode='same')
				r2 = fftconvolve(image_, Psi[2*j+1,...], mode='same')
				inv = np.arctan2(r2,r1)
				M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),-180.*theta/(np.pi),1)
				inv = cv2.warpAffine(inv,M,image.shape,flags=cv2.INTER_CUBIC)
				phase.append(inv[100,100])
			
			# For each image append 
			phase = np.hstack(phase)
			# Find jumps and rectify
			phase_diff = np.append(np.diff(phase),phase[0]-phase[-1])
			argmax = np.argmax(np.abs(phase_diff))
			phase = np.roll(phase, phase.shape[0]-argmax-1)
			# Measure deviation, corrected for frequency 1 curves only
			deviation = phase-Theta+np.pi
			if np.amax(deviation) < 1.:
				deviations.append(np.amax(np.abs(deviation)))
				#plt.plot(Theta, deviation)
		#plt.draw()
		#raw_input()
		max_error.append(np.amax(deviations))
	max_error = np.hstack(max_error)
	plt.plot(N_range, 360.*max_error/(2.*np.pi))
	plt.xlabel('Filter size')
	plt.ylabel('Max angular error (degrees)')
	plt.show()





if __name__ == '__main__':
	magnitude_response()
	#phase_response()
	#view_filters()






























