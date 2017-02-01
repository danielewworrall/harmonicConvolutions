'''Analyse the stability of features under rotation'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import skimage.draw as skdr
import skimage.io as skio
import tensorflow as tf

import harmonic_network_lite as hn_lite

from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


def L2_grid(center, shape):
    # Get neighbourhoods
    lin = np.arange(shape)+0.5
    J, I = np.meshgrid(lin, lin)
    I = I - center[1]
    J = J - center[0]
    dist2 = I**2 + J**2
    return dist2


def main():
    # Load image
    image = skio.imread('../images/scene.jpg')
    image = image[:,266:266+1068,0]
    image = image[500:601,500:601]
    image = image.astype(np.float32)/255.
    '''
    N = 50
    plt.ion()
    plt.show()
    for n in xrange(N):
        angle = (360.*n)/N
        im_rot = rotate(image, angle, reshape=False, order=5)
        im_rot = im_rot[45:56,45:56]*mask
        plt.imshow(im_rot, cmap='gray', interpolation='nearest')
        plt.draw()
        raw_input()
    '''
    # Build network
    x = tf.placeholder(tf.float32, [1,11,11,1,1,1], name='input')
    h = x
    with tf.device('/gpu:0'):
        for j in xrange(1):
            h = hn_lite.conv2d(h, 6, 11, max_order=2, stddev=0.1, name='conv'+str(j), phase=True, device='/gpu:0')
            #h = hn_lite.nonlinearity(h, fnc=tf.nn.relu, name='b'+str(j))
        #y = hn_lite.sum_magnitudes(h)
        y = h
    y_mean = []
    y_base = 0
    diff = []
    
    #plt.ion()
    #plt.show()
    N = 36
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for n in xrange(N):
            angle = (360.*n)/(N-1)
            print angle
            im_rot = rotate(image, angle, reshape=False, order=2)
            im_rot = im_rot[45:56,45:56]
            X = im_rot[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
            Y = sess.run(y, feed_dict={x: X})
            diff.append(np.squeeze(np.sum(Y[0,0,0,1,:,0]**2)))

    print diff
    
    plt.figure()
    plt.plot(np.arange(len(diff)), diff)
    plt.ylim([0, np.amax(diff)*1.1])
    plt.show()
    


if __name__ == '__main__':
    main()















