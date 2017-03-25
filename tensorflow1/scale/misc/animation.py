'''Animation'''

import os
import sys
import time

import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt


def generate_curve(n_frames, mu, c):
	x = []
	momentum_last = 0.2
	y_last = 0
	for i in xrange(n_frames):
		momentum = (1-c)*momentum_last + c*np.random.randn()
		y = y_last + mu*momentum
		momentum_last = momentum
		y_last = y
		x.append(y)
	return x


def make_animation(n_frames):
	x = generate_curve(n_frames, 0.4, 0.1)
	y = generate_curve(n_frames, 0.4, 0.1)
	print max(x)
	
	for i in xrange(n_frames-1,-1,-1):
		plt.clf()
		#pl.rcParams['axes.facecolor'] = '#130721'
		plt.scatter(x[i],y[i],color='#ff5353')
		plt.plot(x[:i+1],y[:i+1],color='#ff5353', alpha=1, linewidth=1)
		plt.plot(x[:i+1],y[:i+1],color='#ff5353', alpha=0.4, linewidth=2)
		plt.plot(x[:i+1],y[:i+1],color='#ff5353', alpha=0.2, linewidth=6)
		plt.xlim([min(x),max(x)])
		plt.ylim([min(y),max(y)])
		#plt.xticks([])
		#plt.yticks([])
		plt.axis('off')
		plt.tight_layout()
		plt.savefig('./trajectory/{:04d}.png'.format(i), transparent=True)
		print i


def make_circle(n_frames):
	t = np.linspace(0.,2.*np.pi,num=n_frames)
	x = np.cos(t)
	y = np.sin(t)
	
	for i in xrange(n_frames):
		plt.clf()
		#pl.rcParams['axes.facecolor'] = '#130721'
		plt.scatter(x[i],y[i],color='#ff5353', s=int(200 + 10*np.cos(3*t[i])))
		plt.plot(x[:i+1],y[:i+1],color='#ff5353', alpha=1, linewidth=1)
		plt.plot(x,y,color='#ff5353', alpha=0.4, linewidth=2)
		plt.plot(x,y,color='#ff5353', alpha=0.2, linewidth=6)
		plt.xlim([-1.1,1.1])
		plt.ylim([-1.1,1.1])
		#plt.xticks([])
		#plt.yticks([])
		plt.axis('off')
		#plt.tight_layout()
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig('./circle/{:04d}.png'.format(i), transparent=True)
		print i


def make_semicircle(n_frames):
	t = np.pi*(0.5*np.cos(np.linspace(0.,2.*np.pi,num=n_frames))+0.5) - np.pi
	s = np.linspace(0,2.*np.pi,num=n_frames)
	x = np.cos(t)
	y = np.sin(t)
	xs = np.cos(s)
	ys = np.sin(s)
	
	for i in xrange(n_frames):
		plt.clf()
		#pl.rcParams['axes.facecolor'] = '#130721'
		plt.scatter(x[i],y[i],color='#ff5353', s=int(200 + 10*np.cos(3*t[i])))
		plt.plot(x[:i+1],y[:i+1],color='#ff5353', alpha=1, linewidth=1)
		plt.plot(xs,ys,color='#ff5353', alpha=0.4, linewidth=2)
		plt.plot(xs,ys,color='#ff5353', alpha=0.2, linewidth=6)
		plt.xlim([-1.1,1.1])
		plt.ylim([-1.1,1.1])
		#plt.xticks([])
		#plt.yticks([])
		plt.axis('off')
		#plt.tight_layout()
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig('./semicircle/{:04d}.png'.format(i), transparent=True)
		print i

if __name__ == '__main__':
	n_frames = 300
	#make_animation(n_frames)
	#make_circle(100)
	make_semicircle(100)