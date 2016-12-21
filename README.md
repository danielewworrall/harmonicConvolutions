# Harmonic Networks: Deep Translation and Rotation Equivariance

> Please read the following information carefully and let us know if anything is missing/you have discovered a bug. We always welcome your feedback!

This folder contains the basic material to construct Harmonic Networks (HNets). Please see our <a href="http://visual.cs.ucl.ac.uk/pubs/harmonicNets/index.html"> project page </a> for more details.
* `train.py` is the main entry point for our code.
* `model_assembly_train.py` contains our multi-gpu training routines that serve as an example of how our functions can interface with regular tensorflow code.
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
python train.py 0 mnist deep_stable <yourDataDirectory>
```
Here, `<yourDataDirectory>` is the folder into which the datasets will be downloaded, the `0` means we will be using GPU 0, `mnist` signifies the dataset to train on, and `deep_stable` the network model as defined in `harmonic_network_models.py`.
You can train on more than one GPU by making the first argument a comma-separated list. For example `0,1,2` would run the training code on the first three GPUs of a system.

Dependencies:
* we require Tensorflow 0.11 as documented <a href="https://www.tensorflow.org/versions/r0.11/api_docs/index.html">here</a>. Newer versions of the API may be supported in future.

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
- [ ] API simplification
- [ ] Longer tutorial

# HNet Tensorflow Tutorial
## Complex Convolutions
The key component of our HNet ops is the complex convolution, which we can approximate using 4 real-valued convolutions. This is implemented in the `complex_conv` function contained in `harmonic_network_ops.py`.
Using CUDNN, we can write this in Tensorflow as follows (where r denotes a real and i an imaginary tensor):

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
This is why we provide a special function for the first layer, `real_input_rotated_conv`, which returns a complex tensor but takes as input a real one, such as an image.
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

## Building Networks of Complex Convolutions

As an example of how to use these functions, we will build the MNIST network from the paper.

```python
def deep_mnist(opt, x, train_phase, device='/cpu:0'):
"""High frequency convolutions are unstable, so get rid of them"""
# Sure layers weight & bias
order = 1
nf = opt['n_filters']
nf2 = int(nf*opt['filter_gain'])
nf3 = int(nf*(opt['filter_gain']**2.))
bs = opt['batch_size']
n = ((opt['filter_size']+1)/2)
tr1 = (n*(n+1))/2
tr2 = tr1 - 1

sm = opt['std_mult']
with tf.device(device):
    weights = {
        'w1' : get_weights_dict([[tr1,],[tr2,]], opt['n_channels'], nf, std_mult=sm, name='W1', device=device),
        'w2' : get_weights_dict([[tr1,],[tr2,]], nf, nf, std_mult=sm, name='W2', device=device),
        'w3' : get_weights_dict([[tr1,],[tr2,]], nf, nf2, std_mult=sm, name='W3', device=device),
        'w4' : get_weights_dict([[tr1,],[tr2,]], nf2, nf2, std_mult=sm, name='W4', device=device),
        'w5' : get_weights_dict([[tr1,],[tr2,]], nf2, nf3, std_mult=sm, name='W5', device=device),
        'w6' : get_weights_dict([[tr1,],[tr2,]], nf3, nf3, std_mult=sm, name='W6', device=device),
        'w7' : get_weights_dict([[tr1,],[tr2,]], nf3, opt['n_classes'], std_mult=sm, name='W7', device=device),
    }
    
    biases = {
        'b1' : get_bias_dict(nf, order, name='b1', device=device),
        'b2' : get_bias_dict(nf, order, name='b2', device=device),
        'b3' : get_bias_dict(nf2, order, name='b3', device=device),
        'b4' : get_bias_dict(nf2, order, name='b4', device=device),
        'b5' : get_bias_dict(nf3, order, name='b5', device=device),
        'b6' : get_bias_dict(nf3, order, name='b6', device=device),
        'b7' : tf.get_variable('b7', dtype=tf.float32, shape=[opt['n_classes']],
            initializer=tf.constant_initializer(1e-2)),
        'psi1' : get_phase_dict(1, nf, order, name='psi1', device=device),
        'psi2' : get_phase_dict(nf, nf, order, name='psi2', device=device),
        'psi3' : get_phase_dict(nf, nf2, order, name='psi3', device=device),
        'psi4' : get_phase_dict(nf2, nf2, order, name='psi4', device=device),
        'psi5' : get_phase_dict(nf2, nf3, order, name='psi5', device=device),
        'psi6' : get_phase_dict(nf3, nf3, order, name='psi6', device=device)
    }
    # Reshape input picture -- square inputs for now
    size = opt['dim'] - 2*opt['crop_shape']
    x = tf.reshape(x, shape=[bs,size,size,opt['n_channels']])

fms = []
# Convolutional Layers
with tf.name_scope('block1') as scope:
    cv1 = real_input_rotated_conv(x, weights['w1'], biases['psi1'],
                                    filter_size=opt['filter_size'], padding='SAME', name='1')
    cv1 = complex_nonlinearity(cv1, biases['b1'], tf.nn.relu)
    fms.append(cv1)	
    # LAYER 2
    cv2 = complex_input_rotated_conv(cv1, weights['w2'], biases['psi2'],
                                        filter_size=opt['filter_size'], output_orders=[0,1],
                                        padding='SAME', name='2')
    cv2 = complex_batch_norm(cv2, tf.nn.relu, train_phase,
                                name='batchNorm1', device=device)
    fms.append(cv2)
with tf.name_scope('block2') as scope:
    cv2 = mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
    # LAYER 3
    cv3 = complex_input_rotated_conv(cv2, weights['w3'], biases['psi3'],
                                        filter_size=opt['filter_size'], output_orders=[0,1],
                                        padding='SAME', name='3')
    cv3 = complex_nonlinearity(cv3, biases['b3'], tf.nn.relu)
    fms.append(cv3)
    # LAYER 4
    cv4 = complex_input_rotated_conv(cv3, weights['w4'], biases['psi4'],
                                        filter_size=opt['filter_size'], output_orders=[0,1],
                                        padding='SAME', name='4')
    cv4 = complex_batch_norm(cv4, tf.nn.relu, train_phase,
                                name='batchNorm2', device=device)
    fms.append(cv4)
with tf.name_scope('block3') as scope:
    cv4 = mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
    # LAYER 5
    cv5 = complex_input_rotated_conv(cv4, weights['w5'], biases['psi5'],
                                        filter_size=opt['filter_size'], output_orders=[0,1],
                                        padding='SAME', name='5')
    cv5 = complex_nonlinearity(cv5, biases['b5'], tf.nn.relu)
    fms.append(cv5)
    # LAYER 6
    cv6 = complex_input_rotated_conv(cv5, weights['w6'], biases['psi6'],
                                        filter_size=opt['filter_size'], output_orders=[0,1],
                                        padding='SAME', name='4')
    cv6 = complex_batch_norm(cv6, tf.nn.relu, train_phase,
                                name='batchNorm3', device=device)
    fms.append(cv6)
# LAYER 7
with tf.name_scope('block4') as scope:
    cv7 = complex_input_conv(cv6, weights['w7'], filter_size=opt['filter_size'],
                                padding='SAME', name='7')
    cv7 = tf.reduce_mean(sum_magnitudes(cv7), reduction_indices=[1,2])
    return tf.nn.bias_add(cv7, biases['b7']) #, fms
```
