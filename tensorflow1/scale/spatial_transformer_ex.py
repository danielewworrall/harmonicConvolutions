# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from scipy import ndimage
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
import matplotlib.pyplot as plt

# %% Create a batch of three images (1600 x 1200)
# %% Image retrieved from:
# %% https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = ndimage.imread('../images/balloons.jpg')
im = im / 255.

im = im.reshape(1, 306,700, 3)
im = im.astype('float32')

# %% Let the output size of the transformer be half the image size.
out_size = (306, 700)

# %% Simulate batch
batch = np.append(im, im, axis=0)
batch = np.append(batch, im, axis=0)
print batch.shape
num_batch = 1

x = tf.placeholder(tf.float32, [1, 306, 700, 3])
#x = tf.cast(batch, 'float32')

def transform(theta, imsh):
	scale1 = np.array([[float(imsh[1])/imsh[2], 0.], [0., 1.]])
	scale2 = np.array([[float(imsh[2])/imsh[1], 0.], [0., 1.]])
	rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	mat = np.dot(scale1, rot)
	mat = np.dot(mat, scale2)
	return np.hstack((mat,np.zeros((2,1))))


# %% Create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):
	
	# %% Create a fully-connected layer with 6 output nodes
	n_fc = 6
	W_fc1 = tf.Variable(tf.zeros([306 * 700 * 3, n_fc]), name='W_fc1')
	
	# %% Zoom into the image
	initial = transform(np.pi/4., im.shape)	
	initial = initial.astype('float32')
	initial = initial.flatten()
	
	b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
	h_fc1 = tf.matmul(tf.zeros([num_batch, 306 * 700 * 3]), W_fc1) + b_fc1
	print h_fc1
	print x
	h_trans = transformer(x, h_fc1, out_size)

# %% Run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y = sess.run(h_trans, feed_dict={x: im})

plt.imshow(y[0])
plt.show()




















