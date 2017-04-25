'''BSD model'''

import os
import sys
import time
sys.path.append('../')

import numpy as np
import tensorflow as tf

import harmonic_network_lite as hl


def to_4d(x):
    """Convert tensor to 4d"""
    xsh = np.asarray(x.get_shape().as_list())
    return tf.reshape(x, [xsh[0], xsh[1], xsh[2], np.prod(xsh[3:])])


def hnet_bsd(args, x, train_phase):
    """High frequency convolutions are unstable, so get rid of them"""
    # Sure layers weight & bias
    order = 1
    nf = int(args.n_filters)
    nf2 = int((args.filter_gain)*nf)
    nf3 = int((args.filter_gain**2)*nf)
    nf4 = int((args.filter_gain**3)*nf)
    bs = args.batch_size
    fs = args.filter_size
    nch = args.n_channels
    nr = args.n_rings
    tp = train_phase
    std = args.std_mult

    x = tf.reshape(x, shape=[bs,args.height,args.width,1,1,3])
    fm = {}

    # Convolutional Layers
    with tf.name_scope('stage1') as scope:
        cv1 = hl.conv2d(x, nf, fs, stddev=std, n_rings=nr, name='1_1')
        cv1 = hl.non_linearity(cv1, name='1_1')

        cv2 = hl.conv2d(cv1, nf, fs, stddev=std, n_rings=nr, name='1_2')
        cv2 = hl.batch_norm(cv2, tp, name='bn1')
        mags = to_4d(hl.stack_magnitudes(cv2))
        fm[1] = linear(mags, 1, 1, name='sw1')

    with tf.name_scope('stage2') as scope:
        cv3 = hl.mean_pooling(cv2, ksize=(1,2,2,1), strides=(1,2,2,1))
        cv3 = hl.conv2d(cv3, nf2, fs, stddev=std, n_rings=nr, name='2_1')
        cv3 = hl.non_linearity(cv3, name='2_1')

        cv4 = hl.conv2d(cv3, nf2, fs, stddev=std, n_rings=nr, name='2_2')
        cv4 = hl.batch_norm(cv4, train_phase, name='bn2')
        mags = to_4d(hl.stack_magnitudes(cv4))
        fm[2] = linear(mags, 1, 1, name='sw2')

    with tf.name_scope('stage3') as scope:
        cv5 = hl.mean_pooling(cv4, ksize=(1,2,2,1), strides=(1,2,2,1))
        cv5 = hl.conv2d(cv5, nf3, fs, stddev=std, n_rings=nr, name='3_1')
        cv5 = hl.non_linearity(cv5, name='3_1')

        cv6 = hl.conv2d(cv5, nf3, fs, stddev=std, n_rings=nr, name='3_2')
        cv6 = hl.batch_norm(cv6, train_phase, name='bn3')
        mags = to_4d(hl.stack_magnitudes(cv6))
        fm[3] = linear(mags, 1, 1, name='sw3')

    with tf.name_scope('stage4') as scope:
        cv7 = hl.mean_pooling(cv6, ksize=(1,2,2,1), strides=(1,2,2,1))
        cv7 = hl.conv2d(cv7, nf4, fs, stddev=std, n_rings=nr, name='4_1')
        cv7 = hl.non_linearity(cv7, name='4_1')

        cv8 = hl.conv2d(cv7, nf4, fs, stddev=std, n_rings=nr, name='4_2')
        cv8 = hl.batch_norm(cv8, train_phase, name='bn4')
        mags = to_4d(hl.stack_magnitudes(cv8))
        fm[4] = linear(mags, 1, 1, name='sw4')

    with tf.name_scope('stage5') as scope:
        cv9 = hl.mean_pooling(cv8, ksize=(1,2,2,1), strides=(1,2,2,1))
        cv9 = hl.conv2d(cv9, nf4, fs, stddev=std, n_rings=nr, name='5_1')
        cv9 = hl.non_linearity(cv9, name='5_1')

        cv10 = hl.conv2d(cv9, nf4, fs, stddev=std, n_rings=nr, name='5_2')
        cv10 = hl.batch_norm(cv10, train_phase, name='bn5')
        mags = to_4d(hl.stack_magnitudes(cv10))
        fm[5] = linear(mags, 1, 1, name='sw5')

    fms = {}
    side_preds = []
    xsh = tf.shape(x)
    with tf.name_scope('fusion') as scope:
        for key in fm.keys():
            fms[key] = tf.image.resize_images(fm[key], tf.stack([xsh[1], xsh[2]]))
            side_preds.append(fms[key])
        side_preds = tf.concat(axis=3, values=side_preds)

        fms['fuse'] = linear(side_preds, 1, 1, bias_init=0.01, name='side_preds')
        return fms


def vgg_bsd(args, x, train_phase):
    """High frequency convolutions are unstable, so get rid of them"""
    # Sure layers weight & bias
    nf = int(args.n_filters)
    nf2 = int((args.filter_gain)*nf)
    nf3 = int((args.filter_gain**2)*nf)
    nf4 = int((args.filter_gain**3)*nf)
    bs = args.batch_size
    fs = args.filter_size
    nch = args.n_channels
    tp = train_phase

    fm = {}
    # Convolutional Layers
    with tf.name_scope('stage1') as scope:
        cv1 = linear(x, nf, fs, name='1_1')
        cv1 = tf.nn.relu(cv1, name='1_1')

        cv2 = linear(cv1, nf, fs, name='1_2')
        cv2 = Zbn(cv2, tp, name='bn1')
        cv2 = tf.nn.relu(cv2)
        fm[1] = linear(cv2, 1, 1, name='fm1')

    with tf.name_scope('stage2') as scope:
        cv3 = tf.nn.max_pool(cv2, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
        cv3 = linear(cv3, nf2, fs, name='2_1')
        cv3 = tf.nn.relu(cv3, name='2_1')

        cv4 = linear(cv3, nf2, fs, name='2_2')
        cv4 = Zbn(cv4, train_phase, name='bn2')
        cv4 = tf.nn.relu(cv4)
        fm[2] = linear(cv4, 1, 1, name='fm2')

    with tf.name_scope('stage3') as scope:
        cv5 = tf.nn.max_pool(cv4, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
        cv5 = linear(cv5, nf3, fs, name='3_1')
        cv5 = tf.nn.relu(cv5, name='3_1')

        cv6 = linear(cv5, nf3, fs, name='3_2')
        cv6 = Zbn(cv6, train_phase, name='bn3')
        cv6 = tf.nn.relu(cv6)
        fm[3] = linear(cv6, 1, 1, name='fm3')

    with tf.name_scope('stage4') as scope:
        cv7 = tf.nn.max_pool(cv6, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
        cv7 = linear(cv7, nf4, fs, name='4_1')
        cv7 = tf.nn.relu(cv7, name='4_1')

        cv8 = linear(cv7, nf4, fs, name='4_2')
        cv8 = Zbn(cv8, train_phase, name='bn4')
        cv8 = tf.nn.relu(cv8)
        fm[4] = linear(cv8, 1, 1, name='fm4')

    with tf.name_scope('stage5') as scope:
        cv9 = tf.nn.max_pool(cv8, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
        cv9 = linear(cv9, nf4, fs, name='5_1')
        cv9 = tf.nn.relu(cv9, name='5_1')

        cv10 = linear(cv9, nf4, fs,  name='5_2')
        cv10 = Zbn(cv10, train_phase, name='bn5')
        cv10 = tf.nn.relu(cv10)
        fm[5] = linear(cv10, 1, 1, name='fm5')

    fms = {}
    side_preds = []
    xsh = tf.shape(x)
    with tf.name_scope('fusion') as scope:
        for key in fm.keys():
            fms[key] = tf.image.resize_images(fm[key], tf.stack([xsh[1], xsh[2]]))
            side_preds.append(fms[key])
        side_preds = tf.concat(axis=3, values=side_preds)

        fms['fuse'] = linear(side_preds, 1, 1, bias_init=0.01, name='fuse')
        return fms


##### LAYERS #####
def linear(x, n_out, ksize, bias_init=None, strides=(1,1,1,1), padding='SAME', name=''):
    """Basic linear matmul layer"""
    xsh = x.get_shape()
    shape = [ksize,ksize,xsh[3],n_out]
    He_initializer = tf.contrib.layers.variance_scaling_initializer()
    W = tf.get_variable(name+'_W', shape=shape, initializer=He_initializer)
    z = tf.nn.conv2d(x, W, strides=strides, padding=padding, name='mul'+str(name))
    if bias_init == None:
        return z
    else:
        return bias_add(z, shape[3], bias_init=bias_init, name=name)


def bias_add(x, nc, bias_init=0.01, name=''):
    const_initializer = tf.constant_initializer(value=bias_init)
    b = tf.get_variable(name+'_b', shape=nc, initializer=const_initializer)
    return tf.nn.bias_add(x, b)


##### SPECIAL FUNCTIONS #####
def Zbn(X, train_phase, decay=0.99, name='batchNorm', device='/cpu:0'):
    """Batch normalization module.

    X: tf tensor
    train_phase: boolean flag True: training mode, False: test mode
    decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
    name: (default batchNorm)

    Source: bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
    batch-normalization-in-tensorflow"""
    n_out = X.get_shape().as_list()[3]

    with tf.name_scope(name) as scope:
        with tf.device(device):
            beta = tf.get_variable(name+'_beta', dtype=tf.float32, shape=n_out,
                                          initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable(name+'_gamma', dtype=tf.float32, shape=n_out,
                                            initializer=tf.constant_initializer(1.0))
            pop_mean = tf.get_variable(name+'_pop_mean', dtype=tf.float32,
                                                shape=n_out, trainable=False)
            pop_var = tf.get_variable(name+'_pop_var', dtype=tf.float32,
                                              shape=n_out, trainable=False)
            batch_mean, batch_var = tf.nn.moments(X, [0,1,2], name=name+'moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
        pop_var_op = tf.assign(pop_var, ema.average(batch_var))

        with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(train_phase, mean_var_with_update,
                lambda: (pop_mean, pop_var))
    normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
    return normed
