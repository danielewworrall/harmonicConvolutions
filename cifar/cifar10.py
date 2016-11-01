import os
import numpy as np


def get_cifar10_data(datadir, trainfn, valfn):

    assert os.path.exists(datadir), 'Datadir does not exist: %s' % datadir

    raw_datadir = os.path.join(datadir, 'cifar-10-batches-py')
    assert os.path.exists(raw_datadir), 'Could not find CIFAR10 data. Please download cifar-10 from ' \
                                        'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz,\n' \
                                        'Extract the data with: tar zxvf cifar-10-python.tar.gz\n' \
                                        'And move the cifar-10-batches-py folder to datadir (' + datadir + ').'

    processed_datadir = os.path.join(datadir, 'preprocessed')
    if not os.path.exists(processed_datadir):
        preprocess_cifar10(datadir)

    train = np.load(os.path.join(processed_datadir, trainfn))
    val = np.load(os.path.join(processed_datadir, valfn))
    train_data = train['data']
    train_labels = train['labels']
    val_data = val['data']
    val_labels = val['labels']

    return train_data, train_labels, val_data, val_labels


def preprocess_cifar10(datadir):
    print 'Preprocessing...'

    # Load batches
    print '   Loading...'
    train_batch_fns = [os.path.join(datadir, 'cifar-10-batches-py', 'data_batch_' + str(i)) for i in range(1, 6)]
    train_batches = [_load_cifar10_batch(fn) for fn in train_batch_fns]
    test_batch = _load_cifar10_batch(os.path.join(datadir, 'cifar-10-batches-py', 'test_batch'))

    # Stack the batches into one big array
    train_data_all = np.vstack([train_batches[i][0] for i in range(len(train_batches))]).astype('float32')
    train_labels_all = np.vstack([train_batches[i][1] for i in range(len(train_batches))]).flatten()
    test_data = test_batch[0].astype('float32')
    test_labels = test_batch[1]

    # Create train / val split of full train set
    train_data = train_data_all[:40000]
    train_labels = train_labels_all[:40000]
    val_data = train_data_all[40000:]
    val_labels = train_labels_all[40000:]

    # Contrast normalize
    print '   Normalizing...'
    train_data_all = normalize(train_data_all)
    train_data = normalize(train_data)
    test_data = normalize(test_data)
    val_data = normalize(val_data)

    # ZCA Whiten
    print '   Computing whitening matrix...'
    train_data_flat = train_data.reshape(train_data.shape[0], -1).T
    train_data_all_flat = train_data_all.reshape(train_data_all.shape[0], -1).T
    test_data_flat = test_data.reshape(test_data.shape[0], -1).T
    val_data_flat = val_data.reshape(val_data.shape[0], -1).T

    pca = PCA(D=train_data_flat, n_components=train_data_flat.shape[1])
    pca_all = PCA(D=train_data_all_flat, n_components=train_data_all_flat.shape[1])

    print '   Whitening data...'
    train_data_flat = pca.transform(D=train_data_flat, whiten=True, ZCA=True)
    train_data = train_data_flat.T.reshape(train_data.shape)

    train_data_all_flat = pca_all.transform(D=train_data_all_flat, whiten=True, ZCA=True)
    train_data_all = train_data_all_flat.T.reshape(train_data_all.shape)

    test_data_flat = pca_all.transform(D=test_data_flat, whiten=True, ZCA=True)
    test_data = test_data_flat.T.reshape(test_data.shape)

    val_data_flat = pca.transform(D=val_data_flat, whiten=True, ZCA=True)
    val_data = val_data_flat.T.reshape(val_data.shape)

    print '   Saving...'
    outputdir = os.path.join(datadir, 'preprocessed')
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    np.savez(os.path.join(outputdir, 'train.npz'),
             data=train_data,
             labels=train_labels)
    np.savez(os.path.join(outputdir, 'train_all.npz'),
             data=train_data_all,
             labels=train_labels_all)
    np.savez(os.path.join(outputdir, 'val.npz'),
             data=val_data,
             labels=val_labels)
    np.savez(os.path.join(outputdir, 'test.npz'),
             data=test_data,
             labels=test_labels)

    print 'Preprocessing complete'


def _load_cifar10_batch(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict['data'].reshape(-1, 3, 32, 32), dict['labels']


def normalize(data, eps=1e-8):
    data -= data.mean(axis=(1, 2, 3), keepdims=True)
    std = np.sqrt(data.var(axis=(1, 2, 3), ddof=1, keepdims=True))
    std[std < eps] = 1.
    data /= std
    return data


class PCA(object):

    def __init__(self, D, n_components):
        self.n_components = n_components
        self.U, self.S, self.m = self.fit(D, n_components)

    def fit(self, D, n_components):
        """
        The computation works as follows:
        The covariance is C = 1/(n-1) * D * D.T
        The eigendecomp of C is: C = V Sigma V.T
        Let Y = 1/sqrt(n-1) * D
        Let U S V = svd(Y),
        Then the columns of U are the eigenvectors of:
        Y * Y.T = C
        And the singular values S are the sqrts of the eigenvalues of C
        We can apply PCA by multiplying by U.T
        """

        # We require scaled, zero-mean data to SVD,
        # But we don't want to copy or modify user data
        m = np.mean(D, axis=1)[:, np.newaxis]
        D -= m
        D *= 1.0 / np.sqrt(D.shape[1] - 1)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        D *= np.sqrt(D.shape[1] - 1)
        D += m
        return U[:, :n_components], S[:n_components], m

    def transform(self, D, whiten=False, ZCA=False,
                  regularizer=10 ** (-5)):
        """
        We want to whiten, which can be done by multiplying by Sigma^(-1/2) U.T
        Any orthogonal transformation of this is also white,
        and when ZCA=True we choose:
         U Sigma^(-1/2) U.T
        """
        if whiten:
            # Compute Sigma^(-1/2) = S^-1,
            # with smoothing for numerical stability
            Sinv = 1.0 / (self.S + regularizer)

            if ZCA:
                # The ZCA whitening matrix
                W = np.dot(self.U,
                           np.dot(np.diag(Sinv),
                                  self.U.T))
            else:
                # The whitening matrix
                W = np.dot(np.diag(Sinv), self.U.T)

        else:
            W = self.U.T

        # Transform
        return np.dot(W, D - self.m)