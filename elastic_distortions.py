'''Elastic distortions as per Simard'''

import cPickle as pkl
import numpy as np
import skimage.io as skio

from matplotlib import pyplot as plt

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    #assert len(image.shape)==2
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    
    dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    im = []
    for i in xrange(image.shape[2]):
        im.append(map_coordinates(image[...,i], indices, order=1).reshape(shape[:2]))
    return np.stack(im, axis=2)


if __name__ == '__main__':
    with open('./data/bsd_pkl_float/train_images.pkl') as fp:
        data = pkl.load(fp)
    plt.ion()
    plt.show()
    for key, im in data.iteritems():
        im_ = elastic_transform(im['x'], 50., 10.)
        #if im['transposed']:
        #    im_ = np.transpose(im_, (1,0,2))
        print np.amin(im_), np.amax(im_)
        plt.imshow(np.squeeze(im_), interpolation='nearest', cmap='gray')
        plt.draw()
        #plt.pause(0.001)
        raw_input(key)