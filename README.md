# Group Convolutions

Convolutional neural networks are designed to operated over the integer-plane Z2. This induces a translational equivariant property in the convolutional representations; that is, linear shifts in the filters, lead to linear shifts in the corresponding feature maps. We extend this to operate over other algebraic structures (transformations), such as rotations and reflections. 

The work centres on one-parameter subgroups of the toroidal group. Such groups are cyclic structures, generalizing the unit circle and so an example of such a cyclic structure would be convolving an image with all the rotations of a kernel. Naturally, it is an computational intractable problem to convolve with an infinite number of kernels, but we are able to exploit the natural structure of the the one-parameter subgroups to yield an analytical representation of any convolution, given a kernel orientation as a function of kernel angle.
