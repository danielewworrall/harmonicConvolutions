# Harmonic Networks: Deep Translation and Rotation Equivariance
> Please read the following information carefully and let us know if anything is missing/you have discovered a bug. We always welcome your feedback!
This folder contains the basic material to construct Harmonic Networks (HNs). Please see our <a href="http://visual.cs.ucl.ac.uk/pubs/harmonicNets/index.html"> project page </a> for more details.
* `train.py` is the main entry point for our code.
* `model_assembly_train.py` contains our multi-gpu trainig routines that serve as an example of how our functions can interface with regular tensorflow code.
* `io_helpers.py` contains our code for downloading, processing and batching datasets.
* `harmonic_network_ops.py` contains core HN implementations.
* `harmonic_network_helpers.py` contains handy functions for using these (such as block definitions).
* `harmonic_network_models.py` contains the model definitions that are necessary to reproduce our results.

All other scripts are for our purposes only and can be safely ignored by the user.

____
> NOTE: The model definitions and core functions are independent of our training code and can be used without it.
____

To run the MNIST example from the paper, navigate to the parent directory of this repo and type:
```python
python train.py 0 mnist deep_stable <yourDataDirectory>
```
Here, `<yourDataDirectory>` is the folder into which the datasets will be downloaded, the `0` means we will be using GPU 0, `mnist` signifies the dataset to train on, and `deep_stable` the network mdoel as defined in `harmonic_network_models.py`.
You can train on more than one GPU by making the first argument a comma-separated list. For example `0,1,2` would run the training code on the first three GPUs of a system.

Dependencies:
* we require at least tensorflow 0.12 as documented <a href="https://www.tensorflow.org/versions/r0.12/api_docs/index.html">here</a>. Newer versions of the API may be supported in future.

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
- [ ] API simplication
- [ ] Longer tutorial

#HN Tensorflow Tutorial

The key component of our HN ops is the complex convolution, which we can approximate using 4 real-valued convolutions. This is implemented in the `complex_conv` function contained in `harmonic_network_ops.py`.
Using CUDNN, we can write this in tensorflow as follows (where r denotes a real and i an imaginary tensor):

```python
Rrr = tf.nn.conv2d(Xr, Qr, strides=strides, padding=padding)
Rii = tf.nn.conv2d(Xi, Qi, strides=strides, padding=padding)
Rri = tf.nn.conv2d(Xr, Qi, strides=strides, padding=padding)
Rir = tf.nn.conv2d(Xi, Qr, strides=strides, padding=padding)
Rr = Rrr - Rii
Ri = Rri + Rir
return Rr, Ri
```

This function is only useful if the input already has a real and an imaginary component. However, in most cases we wish to process a real-valued input such as an image and obtain another real-valued output for our loss function.
This is why we prodive a special function for the first layer, `real_input_rotated_conv`, which returns a complex tensor but takes as input a real one, such as an image.
After the first layer, we use 'complex_input_rotated_conv' wish works on complex tensors. To obtain a real-valued final tensor from the network, we sum the magnitudes as follows:

```python
def sum_magnitudes(X, eps=1e-4):
	"""Sum the magnitudes of each of the complex feature maps in X.
	
	Output U = sum_i |X_i|
	
	X: dict of channels {rotation order: (real, imaginary)}
	eps: regularization since grad |Z| is infinite at zero (default 1e-4)
	"""
	R = []
	for m, r in X.iteritems():
		R.append(tf.sqrt(tf.square(r[0]) + tf.square(r[1]) + eps))
	return tf.add_n(R)
```

The result of this function can then be used with ops such as `reduce_mean` etc. as normal.
We would like to note how these functions demonstrate our way of handling complex numbers as a dictionary of tensors with a real and an imaginary component.