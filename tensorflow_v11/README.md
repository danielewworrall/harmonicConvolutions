# Harmonic Networks: Deep Translation and Rotation Equivariance

> Please read the following information carefully and let us know if anything is missing/you have discovered a bug. We always welcome your feedback!

This folder contains the basic material to construct Harmonic Networks (HNets). Please see our <a href="http://visual.cs.ucl.ac.uk/pubs/harmonicNets/index.html"> project page </a> for more details.
* `harmonic_network_lite.py` contains the new simple HNet API. These functions are most useful for constructing custom networks and do not depend on any special training routine or optimiser.
* `train.py` is the main entry point for our code.
* `model_assembly_train.py` contains our multi-gpu trainig routines that serve as an example of how our functions can interface with regular tensorflow code.
* `io_helpers.py` contains our code for downloading, processing and batching datasets.
* `harmonic_network_ops.py` contains core HNet implementations.
* `harmonic_network_helpers.py` contains handy functions for using these (such as block definitions).
* `harmonic_network_models.py` contains the model definitions that are necessary to reproduce our results.

All other scripts are for our purposes only and can be safely ignored by the user.

____
> NOTE: The model definitions and core functions are independent of our training code and can be used without it.
____

To run the MNIST example from the paper, navigate to the parent directory of this repo and type:
```python
python train.py 0 mnist deep_mnist <yourDataDirectory>
```
Here, `<yourDataDirectory>` is the folder into which the datasets will be downloaded, the `0` means we will be using GPU 0, `mnist` signifies the dataset to train on, and `deep_stable` the network mdoel as defined in `harmonic_network_models.py`.
You can train on more than one GPU by making the first argument a comma-separated list. For example `0,1,2` would run the training code on the first three GPUs of a system.

Dependencies:
* we require tensorflow 0.12 as documented <a href="https://www.tensorflow.org/versions/r0.12/api_docs/index.html">here</a>. Newer versions of the API may be supported in future and older versions are not supported.

Please note that
* this is work in progress, so pull often!
* the API is not yet stable and subject to change (particularly, we are working on improving the ease of use of our convolution functions).

Todos which we have completed:
- [x] API for core HC functions
- [x] Easy rotated MNIST example
- [x] Short tutorial

Todos which we are currently working on:
- [ ] Providing easy training code for our BSD experiments
- [ ] Providing multi-threaded reads for data-feeding
- [x] API simplication
- [x] Longer tutorial

# HNet Tensorflow Tutorial
Athough it is possible to use the low-level API exposed in 'harmonic_network_ops.py', we recommend using the new 'lite' API, which provides an interface close to tensorflow's own. In this section, we will show how to build a model for MNIST and explain how each operation works and what we need to be careful of. We would like to note at the outset that no special optimiser or training procedure is needed if the network is set up correctly.

To get started with reproducing the network from the paper, we import the relevant functions as follows:
```python
import harmonic_network_lite as hn_lite
```
We can then construct an MNIST model as easily as this (We go through this code below):
```python
def deep_mnist(opt, x, train_phase, device='/cpu:0'):
	order = 1
	# Number of Filters
	nf = opt['n_filters']
	nf2 = int(nf*opt['filter_gain'])
	nf3 = int(nf*(opt['filter_gain']**2.))
	bs = opt['batch_size']
	fs = opt['filter_size']
	nch = opt['n_channels']
	ncl = opt['n_classes']
	d = device
	sm = opt['std_mult']

	# Create bias for final layer
	with tf.device(device):
		bias = tf.get_variable('b7', shape=[opt['n_classes']],
							   initializer=tf.constant_initializer(1e-2))
		x = tf.reshape(x, shape=[bs,opt['dim'],opt['dim'],1,1,nch])
	
	# Convolutional Layers with pooling
	with tf.name_scope('block1') as scope:
		cv1 = hn_lite.conv2d(x, nf, fs, padding='SAME', name='1', device=d)
		cv1 = hn_lite.non_linearity(cv1, tf.nn.relu, name='1', device=d)
		
		cv2 = hn_lite.conv2d(cv1, nf, fs, padding='SAME', name='2', device=d)
		cv2 = hn_lite.batch_norm(cv2, train_phase, name='bn1', device=d)

	with tf.name_scope('block2') as scope:
		cv2 = hn_lite.mean_pool(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv3 = hn_lite.conv2d(cv2, nf2, fs, padding='SAME', name='3', device=d)
		cv3 = hn_lite.non_linearity(cv3, tf.nn.relu, name='3', device=d)
		
		cv4 = hn_lite.conv2d(cv3, nf2, fs, padding='SAME', name='4', device=d)
		cv4 = hn_lite.batch_norm(cv4, train_phase, name='bn2', device=d)

	with tf.name_scope('block3') as scope:
		cv4 = hn_lite.mean_pool(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
		cv5 = hn_lite.conv2d(cv4, nf3, fs, padding='SAME', name='5', device=d)
		cv5 = hn_lite.non_linearity(cv5, tf.nn.relu, name='5', device=d)
		
		cv6 = hn_lite.conv2d(cv5, nf3, fs, padding='SAME', name='6', device=d)
		cv6 = hn_lite.batch_norm(cv6, train_phase, name='bn3', device=d)

	# Final Layer
	with tf.name_scope('block4') as scope:
		print('block4')
		cv7 = hn_lite.conv2d(cv6, ncl, fs, padding='SAME', phase=False,
					 name='7', device=d)
		real = hn_lite.sum_mags(cv7)
		cv7 = tf.reduce_mean(real, reduction_indices=[1,2,3,4])
		return tf.nn.bias_add(cv7, bias) 
```
The most important thing to note is what dimensionality each tensor in this graph has. Ignoring the final bias-add, our network has the following structure and data-flow:
_(Note that the dimensions shown below can be understood in words as **[batch_size, num_cols, num_rows, rotation_order, complex, num_channels]**)_


![MNIST H-Net Model](/docs/images/mnist_illustration.png)

...
