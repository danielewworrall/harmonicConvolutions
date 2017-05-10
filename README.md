# Harmonic Networks: Deep Translation and Rotation Equivariance

> This code requires Tensorflow version 1.0

This code accompanies the paper [Harmonic Networks: Deep Translation and Rotation Equivariance](https://arxiv.org/abs/1612.04642) by [Daniel E. Worrall](http://www0.cs.ucl.ac.uk/staff/D.Worrall/), [Stephan J. Garbin](http://stephangarbin.com/), [Daniyar Turmukhambetov](http://www0.cs.ucl.ac.uk/staff/d.turmukhambetov/), and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/G.Brostow/).

[![Watch the video](https://raw.github.com/GabLeRoux/WebMole/master/ressources/WebMole_Youtube_Video.png)](https://www.youtube.com/watch?v=qoWAFBYOtoU&feature=youtu.be)]

# 1 Running the code
To run code for a specific experiment, run the file `run_<myscript>.py` in the relevant folder.

# 2 Using harmonic convolutions in your code
The core functions for harmonic convolutions can be found in ```harmonic_network_ops.py```. However, the best way to use these operations is via ```harmonic_network_lite.py```. This contains the following functions:

- conv2d
- batch_norm
- non_linearity
- mean_pool
- sum_magnitudes
- stack_magnitudes

Each function takes in a 6D tensor with dimensions: minibatch size, height, width, num rotation orders, num complex channels, num channels. For instance, a real tensor with 16 items of height 128 and width 128, 2 rotation orders and 5 channels would have shape [16,128,128,2,1,5]. Whereas a complex tensor with the same parameters would be of shape [16,128,128,2,2,5].

