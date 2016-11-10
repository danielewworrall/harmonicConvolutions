from scipy.ndimage import distance_transform_edt



def preprocess(edge_stack):
    ''' edge_stack is a HxWxC tensor of all the labels '''
    tmp = edge_stack.sum(2) >= 3
    tmp = distance_transform_edt(1.0 - tmp)
    tmp =  np.clip(1.0 - 0.3 * tmp, 0.0, 1.0)
    return tmp[:, :, None]



from scipy.ndimage.filters import gaussian_filter



def get_eigvecs_and_vals(ims_theano):
    '''
    returns the eigenvals and vecs of the rgb vals of a theano image block
    '''
    if ims_theano.ndim == 2:
        rgb_vals = ims_theano
    elif ims_theano.shape[1] == 3:
        rgb_vals = ims_theano.transpose(1, 0, 2, 3).reshape(3, -1).T
    else:
        all_cols = []
        for idx in xrange(len(ims_theano)):
            mask = ims_theano[idx, -1] > 128
            cols = [ims_theano[idx, chan][mask] for chan in range(3)]
            all_cols.append(np.vstack(cols))
        rgb_vals = np.hstack(all_cols).T
        print rgb_vals.shape

    idxs = np.random.choice(rgb_vals.shape[0], 10000, replace=False)
    rgb_vals_subset = rgb_vals[idxs]
    mean_normed_rgb = rgb_vals_subset - rgb_vals_subset.mean(1)[:, None]

    eigvec, eigval, v = np.linalg.svd(mean_normed_rgb.T, full_matrices=False)
    eigval = np.real(eigval)
    eigval /= eigval.sum()

    return eigvec, eigval




class BaseAugmenter(object):

    def edge_randn(self):
        return 6.0 * randn()

    def scale_randn(self, cropped):
        if cropped:
            if rand() > 0.95:
                return 0.5 + 0.05 * randn()
            else:
                return 1.0 + 0.05 * randn()
        else:
            return 1.0 / (0.15 + rand() * 1.0)

    def set_imgs_for_colour(self, X):
        '''
        Takes a theano stack as input, which it uses to compute colour var
        Need to be careful about masks
        '''
        self.rgb_eigvec, self.rgb_eigval = get_eigvecs_and_vals(X)
        return self

    def set_means(self, means):
        '''
        Setting the per-channel means which have been used to scale the data -
        needed for the gamma transform
        '''
        self.means = means

    def _colour_transforms(self, X, mask=None):

        # colour transform
        if rand() > 0.2:
            X = aug.random_colour_transform(
                X, self.rgb_eigval, self.rgb_eigvec, sd=255*0.1, clip=None)

        # blur
        if rand() > 0.8:
            sigma = rand() * 1.0
            for chan in range(3):
                X[chan, :, :] = gaussian_filter(X[chan, :, :], sigma, order=0)
            if mask is not None:
                mask = gaussian_filter(mask, sigma, order=0)

        # gamma transform
        if rand() > 0.:
            gam = randn() * 0.3 + 1.0
            gam = np.clip(gam, 0.2, 1.5)
            tmp = np.clip(X, 0, 255) / 255.0
            if rand() > 0.5 or mask is None:
                tmp = (tmp ** gam)
            else:
                this_mask = np.tile(mask[None, ...] > 0.5, (3, 1, 1))
                tmp[this_mask] = (tmp[this_mask] ** gam)
            X = tmp * 255.0

        return X, mask



class FullAugmenter(BaseAugmenter):

    def __init__(self, out_H, out_W):
        super(BaseAugmenter, self).__init__()
        self.out_H = out_H
        self.out_W = out_W

    def _flips2(self, X):
        # all possible flips and rotations
        if np.random.rand() > 0.5:
            X = X[:, :, ::-1]
        if np.random.rand() > 0.5:
            X = X[:, ::-1, :]
        return X

    def __call__(self, X, max_y, random=True):

        _X = np.zeros((len(X), 3, self.out_H, self.out_W))
        _Y = np.zeros((len(X), max_y, self.out_H, self.out_W))

        for idx in xrange(len(X)):
            im_labs = np.dstack(X[idx])
            tmp = im_labs.transpose(2, 0, 1)
            tmp = self._flips2(tmp)[:, 1:, 1:]
            if tmp.shape[1] == 320: tmp = tmp.transpose(0, 2, 1)

            # RGB stuff
            tmp_im, _ = self._colour_transforms(tmp[:3, :, :].astype(float))
            tmp_im = np.clip(tmp_im, 0, 255)
            _X[idx] = tmp_im

            # Labels stuff
            tmp_labels = tmp[3:, :, :]

            if 0:
                do_replace = tmp_labels.shape[0] < max_y
                idxs = np.random.choice(tmp_labels.shape[0], replace=do_replace, size=max_y)
                _Y[idx] = tmp_labels[idxs]
            else:
                _Y[idx] = tmp_labels

        return (_X).astype(np.float32), (_Y).astype(np.float32)


usage:

aug = FullAugmenter(480, 320)
mbx, mby = aug(X)