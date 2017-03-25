import os
import sys
import time
import glob
sys.path.append('../')
import numpy as np
import tensorflow as tf
import scipy.linalg as splin
#import skimage.io as skio
import scipy.misc
import binvox_rw

### local files ######

import shapenet_loader
import modelnet_loader
import equivariant_loss as el
from spatial_transformer_3d import AffineVolumeTransformer

### local files end ###


################ UTIL #################
def tf_im_summary(prefix, images):
    for j in xrange(min(10, images.get_shape().as_list()[0])):
        desc_str = '%d'%(j) + '_' + prefix
        tf.summary.image(desc_str, images[j:(j+1), :, :, :], max_outputs=1)

def tf_vol_summary(prefix, vols):
    # need to keep 4-dim tensor for tf_im_summary
    #vols = tf.reduce_sum(vols, axis=4)

    vols_d = tf.reduce_sum(vols, axis=1)
    vols_h = tf.reduce_sum(vols, axis=2)
    vols_w = tf.reduce_sum(vols, axis=3)

    vols_d = vols_d / tf.reduce_max(vols_d, axis=[1,2,3], keep_dims=True)
    vols_h = vols_h / tf.reduce_max(vols_h, axis=[1,2,3], keep_dims=True)
    vols_w = vols_w / tf.reduce_max(vols_w, axis=[1,2,3], keep_dims=True)

    tf_im_summary(prefix + '_d', vols_d)
    tf_im_summary(prefix + '_h', vols_h)
    tf_im_summary(prefix + '_w', vols_w)

def save_binvox(cur_vol, name):
    cur_model = binvox_rw.Voxels(
            data=cur_vol, 
            dims=list(cur_vol.shape), 
            translate=[0,0,0], 
            scale=1.0, 
            axis_order='xyz')
    with open(name + '.binvox', 'wb') as f:
        cur_model.write(f)

def imsave(path, img):
    if img.shape[-1]==1:
        img = np.squeeze(img)
    scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(path)

def get_imgs_from_vol(tile_image, tile_h, tile_w):
    tile_image = tile_image.astype(np.float32)
    tile_image = np.sum(tile_image, axis=4)
    tile_image_d = np.sum(tile_image, axis=1)
    tile_image_h = np.sum(tile_image, axis=2)
    tile_image_w = np.sum(tile_image, axis=3)

    d_sum = np.sum(np.sum(tile_image_d, axis=2, keepdims=True), axis=1, keepdims=True)
    h_sum = np.sum(np.sum(tile_image_h, axis=2, keepdims=True), axis=1, keepdims=True)
    w_sum = np.sum(np.sum(tile_image_w, axis=2, keepdims=True), axis=1, keepdims=True)

    d_max = tile_image_d.max(axis=2, keepdims=True).max(axis=1, keepdims=True)
    h_max = tile_image_h.max(axis=2, keepdims=True).max(axis=1, keepdims=True)
    w_max = tile_image_w.max(axis=2, keepdims=True).max(axis=1, keepdims=True)
    
    tile_image_d = tile_image_d / d_max
    tile_image_h = tile_image_h / h_max
    tile_image_w = tile_image_w / w_max

    def tile_batch(batch, tile_w=1, tile_h=1):
        assert tile_w * tile_h == batch.shape[0], 'tile dimensions inconsistent'
        batch = np.split(batch, tile_w*tile_h, axis=0)
        batch = [np.concatenate(batch[i*tile_w:(i+1)*tile_w], 2) for i in range(tile_h)]
        batch = np.concatenate(batch, 1)
        batch = batch[0,:,:]
        return batch

    tile_image_d = tile_batch(tile_image_d, tile_h, tile_w)
    tile_image_h = tile_batch(tile_image_h, tile_h, tile_w)
    tile_image_w = tile_batch(tile_image_w, tile_h, tile_w)

    return tile_image_d, tile_image_h, tile_image_w


def variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False), trainable=True):
    #tf.constant_initializer(0.0)
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def tf_nn_lrelu(x, alpha=0.1, name='lrelu'):
    return tf.nn.relu(x, name=name+'_a') + tf.nn.relu(-alpha*x, name=name+'_b')


    
def convlayer(i, inp, ksize, inpdim, outdim, stride, reuse, nonlin=tf.nn.elu, dobn=True, padding='SAME', is_training=True):
    scopename = 'conv_layer' + str(i)
    print(scopename)
    print(' input:', inp)
    strides = [1, stride, stride, stride, 1]
    with tf.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = variable(scopename + '_kernel', [ksize, ksize, ksize, inpdim, outdim])
        bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
        linout = tf.nn.conv3d(inp, kernel, strides=strides, padding=padding)
        linout = tf.nn.bias_add(linout, bias)
        if dobn:
            bnout = bn5d(linout, is_training, reuse=reuse)
        else:
            bnout = linout
        out = nonlin(bnout, name=scopename + '_nonlin')
    print(' out:', out)
    return out



def vol_resize_nearest(x, outshapes, align_corners=None):
    outdepth, outheight, outwidth = outshapes
    batch_size, depth, height, width, inpdim = x.get_shape().as_list()
    xd_hw = tf.reshape(x, [-1, height, width, inpdim])
    xd_hhww = tf.image.resize_nearest_neighbor(xd_hw, [outheight, outwidth], align_corners=align_corners)
    x_dhhww = tf.reshape(xd_hhww, [batch_size, depth, outheight, outwidth, inpdim])
    x_hhwwd = tf.transpose(x_dhhww, [0,2,3,1,4])
    xhh_wwd = tf.reshape(x_hhwwd, [-1, outwidth, depth, inpdim])
    xhh_wwdd = tf.image.resize_nearest_neighbor(xhh_wwd, [outwidth, outdepth], align_corners=align_corners)
    x_hhwwdd = tf.reshape(xhh_wwdd, [batch_size, outheight, outwidth, outdepth, inpdim])
    x_ddhhww = tf.transpose(x_hhwwdd, [0,3,1,2,4])
    return x_ddhhww


def upconvlayer(i, inp, ksize, inpdim, outdim, outshape, reuse, nonlin=tf.nn.elu, dobn=True, is_training=True):
    scopename = 'upconv_layer' + str(i)
    inpshape = inp.get_shape().as_list()[-2]
    print(scopename)
    print(' input:', inp)
    #pad_size = int(ksize//2)
    #paddings = [[0,0], [pad_size, pad_size], [pad_size, pad_size], [pad_size, pad_size], [0,0]]
    output_shape = [outshape, outshape, outshape]
    padding='SAME'
    with tf.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = variable(scopename + '_kernel', [ksize, ksize, ksize, inpdim, outdim])
        bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
        if outshape>inpshape:
            inp_resized = vol_resize_nearest(inp, output_shape)
        else:
            inp_resized = inp
        #inp_resized_padded = tf.pad(inp_resized, paddings, mode='SYMMETRIC')
        linout = tf.nn.conv3d(inp_resized, kernel, strides=[1, 1, 1, 1, 1], padding=padding)
        linout = tf.nn.bias_add(linout, bias)
        if dobn:
            bnout = bn5d(linout, is_training, reuse=reuse)
        else:
            bnout = linout
        out = nonlin(bnout, name=scopename + 'nonlin')
    print(' out:', out)
    return out

def upconvlayer_tr(i, inp, ksize, inpdim, outdim, outshape, stride, reuse, nonlin=tf.nn.elu, dobn=True, padding='SAME', is_training=True):
    scopename = 'upconv_layer_tr_' + str(i)
    print(scopename)
    print(' input:', inp)
    output_shape = [inp.get_shape().as_list()[0], outshape, outshape, outshape, outdim]
    strides = [1, stride, stride, stride, 1]
    with tf.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = variable(scopename + '_kernel', [ksize, ksize, ksize, outdim, inpdim])
        bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
        linout = tf.nn.conv3d_transpose(inp, kernel, output_shape, strides=strides, padding=padding)
        linout = tf.nn.bias_add(linout, bias)
        if dobn:
            bnout = bn5d(linout, is_training, reuse=reuse)
        else:
            bnout = linout
        out = nonlin(bnout, name=scopename + 'nonlin')
    print(' out:', out)
    return out


def mlplayer(i, inp, inpdim, outdim, is_training, reuse=False, nonlin=tf.nn.elu, dobn=True):
    scopename = 'classifier_' + str(i)
    print(scopename)
    print(' input:', inp)
    with tf.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = variable(scopename + '_kernel', [inpdim, outdim])
        bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
        linout = tf.matmul(inp, kernel)
        linout = tf.nn.bias_add(linout, bias)
        if dobn:
            bnout = bn2d(linout, is_training, reuse=reuse)
        else:
            bnout = linout
        out = nonlin(bnout, name=scopename + '_nonlin')
    print(' out:', out)
    return out

def spatial_pad(x, paddings):
    x_padded = tf.pad(x, paddings, mode='CONSTANT')
    return x_padded

def spatial_transform(stl, x, transmat, paddings=None, threshold=True):
    if not paddings is None:
        x_padded = spatial_pad(x, paddings)
    else:
        x_padded = x

    batch_size = transmat.get_shape().as_list()[0]

    if transmat.get_shape().as_list()[2]==3:
        shiftmat = tf.zeros([batch_size,3,1], dtype=tf.float32)
        transmat_full = tf.concat([transmat, shiftmat], axis=2)
    else:
        transmat_full = transmat
        
    transmat_full = tf.reshape(transmat_full, [batch_size, -1])
    x_in = stl.transform(x_padded, transmat_full)
    if threshold:
        # thresholding
        x_in = tf.floor(0.5 + x_in)
    return x_in

def get_3drotmat(xyzrot):
    assert xyzrot.ndim==2, 'xyzrot must be 2 dimensional array'
    batch_size = xyzrot.shape[0]
    assert xyzrot.shape[1]==3, 'must have rotation angles for x,y and z axii'
    phi = xyzrot[:,0]
    theta = xyzrot[:,1]
    psi = xyzrot[:,2]
    rotmat = np.zeros([batch_size, 3, 3])
    rotmat[:,0,0] = np.cos(theta)*np.cos(psi)
    rotmat[:,0,1] = np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi)
    rotmat[:,0,2] = np.sin(phi)*np.sin(psi) - np.cos(phi)*np.sin(theta)*np.cos(psi)
    rotmat[:,1,0] = -np.cos(theta)*np.sin(psi)
    rotmat[:,1,1] = np.cos(phi)*np.cos(psi) - np.sin(phi)*np.sin(theta)*np.sin(psi)
    rotmat[:,1,2] = np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)
    rotmat[:,2,0] = np.sin(theta)
    rotmat[:,2,1] = -np.sin(phi)*np.cos(theta)
    rotmat[:,2,2] = np.cos(phi)*np.cos(theta)
    return rotmat


def get_3dscalemat(xyzfactor):
    batch_size = xyzfactor.shape[0]
    assert xyzfactor.ndim==2, 'xyzfactor must be a 2 dimensional array'
    assert xyzfactor.shape[1]==3, 'xyzfactor must have scale factor for each axis'
    scalemat = np.zeros([batch_size, 3, 3])
    scalemat[:,0,0] = xyzfactor[:,0]
    scalemat[:,1,1] = xyzfactor[:,1]
    scalemat[:,2,2] = xyzfactor[:,2]
    return scalemat

def get_2drotmat(theta):
    batch_size = theta.shape[0]
    Rot = np.zeros([batch_size, 2, 2])
    Rot[:,0,0] = np.cos(theta)
    Rot[:,0,1] = -np.sin(theta)
    Rot[:,1,0] = np.sin(theta)
    Rot[:,1,1] = np.cos(theta)
    return Rot

def get_2drotscalemat(theta, min_scale, max_scale):
    batch_size = theta.shape[0]
    Rot = np.zeros([batch_size, 2, 2])

    if max_scale>min_scale + 1e-6:
        theta = theta-min_scale
        theta = theta/(max_scale-min_scale)
    theta = np.pi*theta
    Rot[:,0,0] = np.cos(theta)
    Rot[:,0,1] = -np.sin(theta)
    Rot[:,1,0] = np.sin(theta)
    Rot[:,1,1] = np.cos(theta)
    return Rot

def bn2d(X, train_phase, decay=0.99, name='batchNorm', reuse=False):
    assert len(X.get_shape().as_list())==2, 'input to bn2d must be 2d tensor'

    n_out = X.get_shape().as_list()[-1]
    
    beta = tf.get_variable('beta_'+name, dtype=tf.float32, shape=n_out, initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma_'+name, dtype=tf.float32, shape=n_out, initializer=tf.constant_initializer(1.0))
    pop_mean = tf.get_variable('pop_mean_'+name, dtype=tf.float32, shape=n_out, trainable=False)
    pop_var = tf.get_variable('pop_var_'+name, dtype=tf.float32, shape=n_out, trainable=False)

    batch_mean, batch_var = tf.nn.moments(X, [0], name='moments_'+name)
    
    if not reuse:
    	ema = tf.train.ExponentialMovingAverage(decay=decay)
    
    	def mean_var_with_update():
    		ema_apply_op = ema.apply([batch_mean, batch_var])
    		pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
    		pop_var_op = tf.assign(pop_var, ema.average(batch_var))
    
    		with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
    			return tf.identity(batch_mean), tf.identity(batch_var)
    	
    	mean, var = tf.cond(train_phase, mean_var_with_update,
    				lambda: (pop_mean, pop_var))
    else:
    	mean, var = tf.cond(train_phase, lambda: (batch_mean, batch_var),
    			lambda: (pop_mean, pop_var))
    	
    return tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-5)



def bn5d(X, train_phase, decay=0.99, name='batchNorm', reuse=False):
    assert len(X.get_shape().as_list())==5, 'input to bn5d must be 5d tensor'

    n_out = X.get_shape().as_list()[-1]
    
    beta = tf.get_variable('beta_'+name, dtype=tf.float32, shape=n_out, initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma_'+name, dtype=tf.float32, shape=n_out, initializer=tf.constant_initializer(1.0))
    pop_mean = tf.get_variable('pop_mean_'+name, dtype=tf.float32, shape=n_out, trainable=False)
    pop_var = tf.get_variable('pop_var_'+name, dtype=tf.float32, shape=n_out, trainable=False)

    batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2, 3], name='moments_'+name)
    
    if not reuse:
    	ema = tf.train.ExponentialMovingAverage(decay=decay)
    
    	def mean_var_with_update():
    		ema_apply_op = ema.apply([batch_mean, batch_var])
    		pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
    		pop_var_op = tf.assign(pop_var, ema.average(batch_var))
    
    		with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
    			return tf.identity(batch_mean), tf.identity(batch_var)
    	
    	mean, var = tf.cond(train_phase, mean_var_with_update,
    				lambda: (pop_mean, pop_var))
    else:
    	mean, var = tf.cond(train_phase, lambda: (batch_mean, batch_var),
    			lambda: (pop_mean, pop_var))
    	
    return tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-5)


