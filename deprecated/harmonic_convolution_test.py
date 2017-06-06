#
#  test by Nate Thomas, 4/13/17
#
#  to be run in https://github.com/deworrall92/harmonicConvolutions
#
#  Notes:  The harmonic network works well for small numbers of layers,
#          but when the stride is greater than 1 or the number of layers
#          is greater than 5 or so, global rotation invariance
#          starts to fail.
#

import sys
sys.path.append('../')

import numpy as np
from scipy.ndimage import rotate
from skimage.color import rgb2gray
from skimage.io import imread, imsave

#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import harmonic_network_lite as hn_lite

BATCH_SIZE = 1
WIDTH = 483
NCHANNELS = 1
INPUT_SHAPE = [BATCH_SIZE, WIDTH, WIDTH, NCHANNELS]

circle_degrees = np.linspace(0., 360., num=BATCH_SIZE, endpoint=False)
'''
def rotate_batch(batch):
    batch_size = batch.shape[0]
    return np.stack([rotate(batch[i],  circle_degrees[i],
                            axes=(1,0), reshape=False, order=3) for i in range(batch_size)])


def unrotate_batch(batch):
    batch_size = batch.shape[0]
    return np.stack([rotate(batch[i], -circle_degrees[i],
                            axes=(1,0), reshape=False, order=3) for i in range(batch_size)])


def batch_plot(inputs, vmin=0., vmax=1., title=None):
    batch_size = inputs.shape[0]
    unit_fig = plt.figure(figsize=(20, 2))
    if title != None:
        unit_fig.suptitle(title)
    for i in range(batch_size):
        plt.subplot(1, batch_size, 1+i)
        image_plot = inputs[i]
        rank = len(image_plot.shape)
        if rank > 2:
            image_plot = image_plot.squeeze(axis=(2,))
        plt.imshow(image_plot,vmin=vmin,vmax=vmax,interpolation='nearest')
'''

#MNIST = input_data.read_data_sets('data/', one_hot=True)
#IMAGE = np.reshape(MNIST.train.next_batch(1)[0][0], [WIDTH, WIDTH, NCHANNELS])
#IMAGE_BATCH = np.stack([IMAGE]*BATCH_SIZE)
IMAGE = rgb2gray(imread('./14092.jpg'))/10.
EMBEDDED = np.zeros((483,483))
EMBEDDED[81:402,1:482]=IMAGE
#IMAGE_BATCH = np.stack([EMBEDDED]*BATCH_SIZE)[...,np.newaxis]
#print IMAGE_BATCH.shape

#ROTATED_INPUT = rotate_batch(IMAGE_BATCH)

LAYER_COUNT = 5
STRIDE = 1

KERNEL_WIDTH = 4
#assert KERNEL_WIDTH % 2 == 1
MAX_ORDER = 1

# Unit test -- Convolution
tf.reset_default_graph()

input_images = tf.placeholder(tf.float32, shape=INPUT_SHAPE)
reshaped_input = tf.reshape(input_images, [BATCH_SIZE, WIDTH, WIDTH, 1, 1, NCHANNELS])
hconv = reshaped_input

for layer_index in range(LAYER_COUNT):
    #if layer_index != 0:
    #    hconv = hn_lite.mean_pool(hconv, ksize=(1,2,2,1), strides=(1,2,2,1))
    hconv = hn_lite.conv2d(hconv, 1, KERNEL_WIDTH, padding='SAME', strides=(1,STRIDE,STRIDE,1),
                           max_order=MAX_ORDER, name='hc'+str(layer_index))
real = hn_lite.sum_magnitudes(hconv)
output_images = tf.nn.tanh(tf.reduce_mean(real, reduction_indices=[3,4]))


with tf.Session() as sess:
    global_init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    sess.run([global_init_op, local_init_op])
    for i in xrange(360):
        input_image = rotate(EMBEDDED, i, axes=(1,0), reshape=False, order=3)[np.newaxis,:,:,np.newaxis]
        real_output = sess.run(output_images, feed_dict={input_images: input_image})[0,:,:,0]
        real_output = rotate(real_output, -i, axes=(1,0), reshape=False, order=3) / 1.2
        print np.amin(real_output), np.amax(real_output)
        imsave('./nathan/activations/nate_{:d}_/{:04d}.jpg'.format(LAYER_COUNT,i), real_output)

'''
print('no fault')
batch_plot(unrotate_batch(ROTATED_INPUT), title="input")
batch_plot(unrotate_batch(real_output), title="output")
plt.show()
'''
