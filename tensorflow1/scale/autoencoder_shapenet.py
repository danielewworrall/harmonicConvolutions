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

### local files ######

import shapenet_loader
import equivariant_loss as el
from spatial_transformer_3d import AffineVolumeTransformer

### local files end ###

################ DATA #################

#-----------ARGS----------
flags = tf.app.flags
FLAGS = flags.FLAGS
#execution modes
flags.DEFINE_boolean('ANALYSE', False, 'runs model analysis')
flags.DEFINE_integer('eq_dim', -1, 'number of latent units to rotate')
flags.DEFINE_float('l2_latent_reg', 1e-6, 'Strength of l2 regularisation on latents')
flags.DEFINE_integer('save_step', 10, 'Interval (epoch) for which to save')
flags.DEFINE_boolean('Daniel', False, 'Daniel execution environment')
flags.DEFINE_boolean('Sleepy', False, 'Sleepy execution environment')
flags.DEFINE_boolean('Dopey', False, 'Dopey execution environment')
flags.DEFINE_boolean('DaniyarSleepy', True, 'Dopey execution environment')
##---------------------

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

################ DATA #################

def load_data():
    shapenet = shapenet_loader.read_data_sets('~/scratch/Datasets/ShapeNetVox32', one_hot=True)
    return shapenet


def checkFolder(dir):
    """Checks if a folder exists and creates it if not.
    dir: directory
    Returns nothing
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def removeAllFilesInDirectory(directory, extension):
    cwd = os.getcwd()
    os.chdir(directory)
    filelist = glob.glob('*' + extension)
    for f in filelist:
        os.remove(f)
    os.chdir(cwd)


############## MODEL ####################

def autoencoder(x, num_latents, f_params, is_training, reuse=False):
    """Build a model to rotate features"""
    with tf.variable_scope('mainModel', reuse=reuse) as scope:
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            codes = encoder(x, num_latents, is_training, reuse=reuse)
        with tf.variable_scope("feature_transformer", reuse=reuse) as scope:
            code_shape = codes.get_shape()
            batch_size = code_shape.as_list()[0]
            codes = tf.reshape(codes, [batch_size, -1])
            codes_transformed = el.feature_transform_matrix_n(codes, codes.get_shape(), f_params)
            codes_transformed = tf.reshape(codes_transformed, code_shape)
        with tf.variable_scope("decoder", reuse=reuse) as scope:
            recons = decoder(codes_transformed, is_training, reuse=reuse)
    return recons, codes


def variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False), trainable=True):
    #tf.random_normal_initializer(stddev=0.0001)
    #tf.constant_initializer(0.0)
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def encoder(x, num_latents, is_training, reuse=False):
    """Encoder with conv3d"""
    
    def convlayer(i, inp, ksize, inpdim, outdim, stride, reuse, nonlin=tf.nn.elu, dobn=True, padding='SAME'):
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

    # TODO increase ksize
    l0 = convlayer(0, x,  3, 1,     16,  1, reuse)                     
    l1 = convlayer(1, l0, 3, 16,    32,  2, reuse)                     
    l2 = convlayer(2, l1, 3, 32,    64,  2, reuse)                     
    l3 = convlayer(3, l2, 3, 64,   128,  2, reuse)                     
    l4 = convlayer(4, l3, 3, 128,  256,  2, reuse)                     
    l5 = convlayer(5, l4, 4, 256,  2**11, 1, reuse, padding='VALID')   
    codes = convlayer(6, l5, 1, 2**11, num_latents, 1, reuse, nonlin=tf.identity, dobn=False) # 1 -> 1
    return codes


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


def decoder(codes, is_training, reuse=False):
    num_latents = codes.get_shape()[-1]

    def upconvlayer(i, inp, ksize, inpdim, outdim, outshape, reuse, nonlin=tf.nn.elu, padding='SAME'):
        scopename = 'upconv_layer' + str(i)
        print(scopename)
        print(' input:', inp)
        output_shape = [inp.get_shape().as_list()[0], outshape, outshape, outshape, outdim]
        with tf.variable_scope(scopename) as scope:
            if reuse:
                scope.reuse_variables()
            kernel = variable(scopename + '_kernel', [ksize, ksize, ksize, inpdim, outdim])
            bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
            inp_resized = vol_resize_nearest(inp, [outshape, outshape, outshape])
            linout = tf.nn.conv3d(inp_resized, kernel, strides=[1, 1, 1, 1, 1], padding=padding)
            linout = tf.nn.bias_add(linout, bias)
            bnout = bn5d(linout, is_training, reuse=reuse)
            out = nonlin(bnout, name=scopename + 'nonlin')
        print(' out:', out)
        return out

    def upconvlayer_tr(i, inp, ksize, inpdim, outdim, outshape, stride, reuse, nonlin=tf.nn.elu, padding='SAME'):
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
            bnout = bn5d(linout, is_training, reuse=reuse)
            out = nonlin(bnout, name=scopename + 'nonlin')
        print(' out:', out)
        return out

    # TODO increase ksize
    #l0 = upconvlayer(0,     codes, 1, num_latents, 512, 1, 1, reuse)
    #l1 = upconvlayer(1,     codes, 1, num_latents, 2048, 1, 1, reuse) #  1 -> 2
    #l2 = upconvlayer(2,     l1,    4, 2048,        512,  4, 4, reuse) #  2 -> 4
    #l3 = upconvlayer(3,     l2,    4, 512,         256,  7, 2, reuse) #  4 -> 7
    #l4 = upconvlayer(4,     l3,    5, 256,         128, 14, 2, reuse) #  7 -> 14
    #l5 = upconvlayer(5,     l4,    5, 128,         64,  28, 2, reuse) # 14 -> 28
    #l6 = upconvlayer(6,     l5,    5, 64,          32,  56, 2, reuse) # 28 -> 56
    #recons = upconvlayer(7, l6,    3, 32,          1,   56, 1, reuse, nonlin=tf.nn.sigmoid)

    l1 = upconvlayer(1,     codes, 1, num_latents, 2**11, 1, reuse) 
    l2 = upconvlayer_tr(2,  l1,    4, 2**11,       512,   4, 4, reuse)
    l3 = upconvlayer(3,     l2,    3, 512,         256,   8, reuse)
    l4 = upconvlayer(4,     l3,    3, 256,         128,  16, reuse)
    l5 = upconvlayer(5,     l4,    3, 128,         64,   32, reuse)
    l6 = upconvlayer(6,     l5,    3, 64,          32,   32, reuse)
    l7 = upconvlayer(7,     l6,    3, 32,          16,   64, reuse)
    recons = upconvlayer(8, l7,    5, 16,          1,    64, reuse, nonlin=tf.nn.sigmoid)
    return recons


def bernoulli_xentropy(x, test_recon):
    """Cross-entropy for Bernoulli variables"""
    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=test_recon)
    return tf.reduce_mean(tf.reduce_sum(x_entropy, axis=(1,2)))


def spatial_transform(stl, x, transmat, paddings):
    x_padded = tf.pad(x, paddings, mode='CONSTANT')
    batch_size = transmat.get_shape().as_list()[0]
    shiftmat = tf.zeros([batch_size,3,1], dtype=tf.float32)
    transmat_full = tf.concat([transmat, shiftmat], axis=2)
    transmat_full = tf.reshape(transmat_full, [batch_size, -1])
    x_in = stl.transform(x_padded, transmat_full)
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

def random_transmats(batch_size):
    """ Random rotations in 3D
    """
    min_scale = 1.0
    max_scale = 1.0

    if False:
        params_inp_rot = np.pi*2*(np.random.rand(batch_size, 3)-0.5)
        params_inp_rot[:,[0]] = 0.0
        params_inp_scale = 1.0 + 0.0*np.random.rand(batch_size, 3)

        params_trg_rot = np.pi*2*(np.random.rand(batch_size, 3)-0.5)
        params_trg_rot[:,[0]] = 0.0
        params_trg_scale = 1.0 + 0.0*np.random.rand(batch_size, 3)
    else:
        params_inp_rot = np.pi*2*(np.random.rand(batch_size, 3)-0.5)
        params_inp_rot[:,1] = params_inp_rot[:,1]/2
        params_inp_scale = min_scale + (max_scale-min_scale)*np.random.rand(batch_size, 3)

        params_trg_rot = np.pi*2*(np.random.rand(batch_size, 3)-0.5)
        params_trg_rot[:,1] = params_trg_rot[:,1]/2
        params_trg_scale = min_scale + (max_scale-min_scale)*np.random.rand(batch_size, 3)

    inp_3drotmat = get_3drotmat(params_inp_rot)
    inp_3dscalemat = get_3dscalemat(params_inp_scale)

    trg_3drotmat = get_3drotmat(params_trg_rot)
    trg_3dscalemat = get_3dscalemat(params_trg_scale)

    # TODO scale and rot because inverse warp in stl
    stl_transmat_inp = np.matmul(inp_3dscalemat, inp_3drotmat)
    stl_transmat_trg = np.matmul(trg_3dscalemat, trg_3drotmat)
    
    f_params_inp = np.zeros([batch_size, 3, 3])
    # TODO was like this:
    #cur_rotmat = np.matmul(trg_3drotmat, inp_3drotmat.transpose([0,2,1]))
    cur_rotmat = np.matmul(trg_3drotmat.transpose([0,2,1]), inp_3drotmat)
    f_params_inp = set_f_params_rot(f_params_inp, cur_rotmat)
    #print(f_params_inp[0,:,:])

    # TODO
    #for i in xrange(3):
    #    inp_f_2dscalemat = get_2drotscalemat(params_inp_scale[:, i], min_scale, max_scale)
    #    trg_f_2dscalemat = get_2drotscalemat(params_trg_scale[:, i], min_scale, max_scale)
    #    cur_f_scalemat = np.matmul(trg_f_2dscalemat, inp_f_2dscalemat.transpose([0,2,1]))
    #    f_params_inp = set_f_params_scale(f_params_inp, i, cur_f_scalemat)

    return stl_transmat_inp.astype(np.float32), stl_transmat_trg.astype(np.float32), f_params_inp.astype(np.float32)

def set_f_params_rot(f_params, rotmat):
    f_params[:,0:3,0:3] = rotmat
    return f_params

def set_f_params_scale(f_params, i, rotmat):
    f_params[:,3+i*2:5+i*2,3+i*2:5+i*2] = rotmat
    return f_params

def mul_f_params_rot(f_params, rotmat):
    f_params[:,0:3,0:3] = np.matmul(rotmat, f_params[:,0:3,0:3])
    return f_params

def mul_f_params_scale(f_params, i, rotmat):
    f_params[:,3+i*2:5+i*2,3+i*2:5+i*2] = np.matmul(rotmat, f_params[:,3+i*2:5+i*2,3+i*2:5+i*2])
    return f_params

def update_f_params(f_params, rot_or_scale, ax, theta):
    if rot_or_scale==0:
        # do 3x3 rotation
        angles = np.zeros([1,3])
        angles[:,ax] = theta
        inp_3drotmat = get_3drotmat(angles)
        f_params = set_f_params_rot(f_params, inp_3drotmat)
    elif rot_or_scale==1:
        # do 2x2 scale rotation
        angles = np.zeros([1,1])
        angles[:,0] = theta
        # TODO min_scale, max_scale
        inp_f_2dscalemat = get_2drotscalemat(angles, 1.0, 1.0)
        f_params = set_f_params_scale(f_params, ax, inp_f_2dscalemat)
    return f_params


##### SPECIAL FUNCTIONS #####
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


############################################

def train(inputs, outputs, ops, opt, data):
    """Training loop"""
    # Unpack inputs, outputs and ops
    x, global_step, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training = inputs
    loss, merged, test_recon, recons = outputs
    train_op = ops
    
    # For checkpoints
    gs = 0
    start = time.time()
    
    saver = tf.train.Saver()
    sess = tf.Session()
    
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    train_writer = tf.summary.FileWriter(opt['summary_path'], sess.graph)

    # Training loop
    for epoch in xrange(opt['n_epochs']):
        # Learning rate
        exponent = sum([epoch > i for i in opt['lr_schedule']])
        current_lr = opt['lr']*np.power(0.2, exponent)
        
        # Train
        train_loss = 0.
        train_acc = 0.
        # Run training steps
        num_steps = data.train.num_steps(opt['mb_size'])
        for step_i in xrange(num_steps):
            cur_stl_params_in, cur_stl_params_trg, cur_f_params = random_transmats(opt['mb_size'])
            ops = [global_step, loss, merged, train_op]
            cur_x, _ = data.train.next_batch(opt['mb_size'])
            
            feed_dict = {
                        x: cur_x,
                        stl_params_trg: cur_stl_params_trg, 
                        stl_params_in: cur_stl_params_in, 
                        f_params : cur_f_params, 
                        lr : current_lr,
                        is_training : True
                    }

            gs, l, summary, __ = sess.run(ops, feed_dict=feed_dict)
            train_loss += l

            # Summary writers
            train_writer.add_summary(summary, gs)

        if epoch % FLAGS.save_step == 0:
            cur_recons = sess.run(recons, feed_dict=feed_dict)
            tile_size = int(np.floor(np.sqrt(cur_recons.shape[0])))
            cur_recons = cur_recons[0:tile_size*tile_size, :,:,:,:]
            tile_image_d, tile_image_h, tile_image_w = get_imgs_from_vol(cur_recons, tile_size, tile_size)
            save_name = './samples/' + opt['flag'] + '/train_image_%04d' % epoch
            imsave(save_name + '_d.png', tile_image_d) 
            imsave(save_name + '_h.png', tile_image_h) 
            imsave(save_name + '_w.png', tile_image_w) 

        train_loss /= num_steps
        print('[{:03d}]: {:03f}'.format(epoch, train_loss))

        # Save model
        if epoch % FLAGS.save_step == 0:
            path = saver.save(sess, opt['save_path'], epoch)
            print('Saved model to ' + path)
        
        # Validation
        if epoch % 2 == 0:
            val_recons = []
            max_angles = 20
            #pick a random initial transformation
            cur_stl_params_in, _, cur_f_params = random_transmats(1)
            cur_x, _ = data.validation.next_batch(1)
            fangles = np.linspace(0., np.pi, num=max_angles)
            fscales = np.linspace(0.8, 1.0, num=max_angles)

            rot_ax = 1#np.random.randint(0, 3)
            for j in xrange(max_angles):
                cur_f_params_j = update_f_params(cur_f_params, 0, rot_ax, fangles[j])
                do_scale_ax = np.random.rand(3)>0.5
                for i in xrange(max_angles):
                    cur_f_params_ji = cur_f_params_j
                    # TODO
                    #for scale_ax in xrange(3):
                    #    if do_scale_ax[scale_ax]:
                    #        cur_f_params_ji = update_f_params(cur_f_params_ji, 1, scale_ax, fscales[i])

                    feed_dict = {
                                test_x : cur_x,
                                test_stl_params_in : cur_stl_params_in, 
                                val_f_params: cur_f_params_ji,
                                is_training : False
                            }

                    y = sess.run(test_recon, feed_dict=feed_dict)
                    val_recons.append(y[0,:,:,:,:].copy())
            
            samples_ = np.stack(val_recons)

            tile_image = np.reshape(samples_, [max_angles*max_angles, opt['outsize'][0], opt['outsize'][1], opt['outsize'][2], opt['color_chn']])

            tile_image_d, tile_image_h, tile_image_w = get_imgs_from_vol(tile_image, max_angles, max_angles)

            save_name = './samples/' + opt['flag'] + '/image_%04d' % epoch
            imsave(save_name + '_d.png', tile_image_d) 
            imsave(save_name + '_h.png', tile_image_h) 
            imsave(save_name + '_w.png', tile_image_w) 

            # TODO save binvox


def main(_):
    opt = {}
    """Main loop"""
    tf.reset_default_graph()
    if FLAGS.Daniel:
        print('Hello Daniel!')
        opt['root'] = '/home/daniel'
        dir_ = opt['root'] + '/Code/harmonicConvolutions/tensorflow1/scale'
    elif FLAGS.Sleepy:
        print('Hello dworrall!')
        opt['root'] = '/home/dworrall'
        dir_ = opt['root'] + '/Code/harmonicConvolutions/tensorflow1/scale'
    elif FLAGS.Dopey:
        print('Hello Daniyar!')
        opt['root'] = '/home/daniyar'
        dir_ = opt['root'] + '/deep_learning/harmonicConvolutions/tensorflow1/scale'
    elif FLAGS.DaniyarSleepy:
        print('Hello Daniyar!')
        opt['root'] = '/home/daniyar'
        dir_ = opt['root'] + '/deep_learning/harmonicConvolutions/tensorflow1/scale'
    else:
        opt['root'] = '/home/sgarbin'
        dir_ = opt['root'] + '/Projects/harmonicConvolutions/tensorflow1/scale'
    
    opt['mb_size'] = 32
    opt['n_epochs'] = 200
    opt['lr_schedule'] = [30, 60, 100, 150, 190]
    opt['lr'] = 1e-3

    opt['vol_size'] = [32,32,32]
    pad_size = 16#int(np.ceil(np.sqrt(3)*opt['vol_size'][0]/2)-opt['vol_size'][0]/2)
    opt['outsize'] = [i + 2*pad_size for i in opt['vol_size']]
    stl = AffineVolumeTransformer(opt['outsize'])
    opt['color_chn'] = 1
    opt['stl_size'] = 3 # no translation
    # TODO
    opt['f_params_dim'] = 3# + 2*3 # rotation matrix is 3x3 and we have 3 axis scalings implemented as 2x2 rotations
    opt['num_latents'] = opt['f_params_dim']*128


    opt['flag'] = 'shapenet'
    opt['summary_path'] = dir_ + '/summaries/autotrain_{:s}'.format(opt['flag'])
    opt['save_path'] = dir_ + '/checkpoints/autotrain_{:s}/model.ckpt'.format(opt['flag'])
    
    #check and clear directories
    checkFolder(opt['summary_path'])
    checkFolder(opt['save_path'])
    checkFolder(dir_ + '/samples/' + opt['flag'])
    #removeAllFilesInDirectory(opt['summary_path'], '.*')
    #removeAllFilesInDirectory(opt['save_path'], '.*')
    
    # Load data
    data = load_data()
    
    # Placeholders
    # batch_size, depth, height, width, in_channels
    x = tf.placeholder(tf.float32, [opt['mb_size'],opt['vol_size'][0],opt['vol_size'][1],opt['vol_size'][2], opt['color_chn']], name='x')
    stl_params_in  = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],opt['stl_size']], name='stl_params_in')
    stl_params_trg = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],opt['stl_size']], name='stl_params_trg')
    f_params = tf.placeholder(tf.float32, [opt['mb_size'], opt['f_params_dim'], opt['f_params_dim']], name='f_params')

    test_x = tf.placeholder(tf.float32, [1,opt['vol_size'][0],opt['vol_size'][1],opt['vol_size'][2],opt['color_chn']], name='test_x')
    test_stl_params_in  = tf.placeholder(tf.float32, [1,opt['stl_size'],opt['stl_size']], name='test_stl_params_in')
    val_f_params = tf.placeholder(tf.float32, [1, opt['f_params_dim'], opt['f_params_dim']], name='val_f_params') 
    paddings = tf.convert_to_tensor(np.array([[0,0], [pad_size,pad_size], [pad_size,pad_size], [pad_size, pad_size], [0,0]]), dtype=tf.int32)
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.placeholder(tf.float32, [], name='lr')
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    
    # Build the training model
    x_in = spatial_transform(stl, x, stl_params_in, paddings)
    x_trg = spatial_transform(stl, x, stl_params_trg, paddings)
    recons, codes = autoencoder(x_in, opt['num_latents'], f_params, is_training)
    
    # Test model
    test_x_in = spatial_transform(stl, test_x, test_stl_params_in, paddings)
    test_recon, __ = autoencoder(test_x_in, opt['num_latents'], val_f_params, is_training, reuse=True)
    
    # LOSS
    loss = bernoulli_xentropy(x_trg, recons)
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_trg-recons), axis=[1,2,3,4]))
    
    # Summaries
    tf_vol_summary('recons', recons) 
    tf_vol_summary('inputs', x_in) 
    tf_vol_summary('targets', x_trg) 
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('LearningRate', lr)
    merged = tf.summary.merge_all()
    
    # Build optimizer
    optim = tf.train.AdamOptimizer(lr)
    train_op = optim.minimize(loss, global_step=global_step)
    
    # Set inputs, outputs, and training ops
    inputs = [x, global_step, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training]
    outputs = [loss, merged, test_recon, recons]
    ops = [train_op]
    
    # Train
    return train(inputs, outputs, ops, opt, data)

if __name__ == '__main__':
    tf.app.run()
