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

from vol_util import *
import shapenet_loader
import modelnet_loader
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
flags.DEFINE_integer('save_step', 2, 'Interval (epoch) for which to save')
flags.DEFINE_boolean('Daniel', False, 'Daniel execution environment')
flags.DEFINE_boolean('Sleepy', False, 'Sleepy execution environment')
flags.DEFINE_boolean('Dopey', False, 'Dopey execution environment')
flags.DEFINE_boolean('DaniyarSleepy', True, 'Dopey execution environment')
flags.DEFINE_boolean('TEST', False, 'Evaluate model on the test set')
flags.DEFINE_boolean('VIS', False, 'Visualize feature space transformations')

##---------------------

################ DATA #################

def load_data():
    #shapenet = shapenet_loader.read_data_sets_splits('~/scratch/Datasets/ShapeNetVox32', one_hot=True)
    #shapenet = shapenet_loader.read_data_sets('~/scratch/Datasets/ShapeNetVox32', one_hot=True)
    #return shapenet
    modelnet = modelnet_loader.read_data_sets('~/scratch/Datasets/ModelNet', one_hot=True)
    return modelnet


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
            recons, recons_logits = decoder(codes_transformed, is_training, reuse=reuse)
    return recons, codes, recons_logits


def encoder(x, num_latents, is_training, reuse=False):
    """Encoder with conv3d"""
    l0 = convlayer(0, x,  3, 1,     8,  1, reuse, is_training=is_training) # 32
    l1 = convlayer(1, l0, 3, 8,     16,  2, reuse, is_training=is_training) # 16
    l2 = convlayer(2, l1, 3, 16,    32,  1, reuse, is_training=is_training) # 16
    l3 = convlayer(3, l2, 3, 32,    64,  2, reuse, is_training=is_training) # 8
    l4 = convlayer(4, l3, 8, 64,   512,  1, reuse, padding='VALID', is_training=is_training)   
    # TODO was using batch norm previously. -_-
    #codes = convlayer(5, l4, 1, 512, num_latents, 1, reuse, nonlin=tf.identity) # 1 -> 1
    codes = convlayer(5, l4, 1, 512, num_latents, 1, reuse, nonlin=tf.identity, dobn=False) # 1 -> 1
    return codes


def decoder(codes, is_training, reuse=False):
    num_latents = codes.get_shape()[-1]

    # Submission architecture
    #l1 = upconvlayer(1,     codes, 1,   num_latents, 512,   1, reuse, is_training=is_training) 
    #l2 = upconvlayer_tr(2,  l1,    8,   512,         128,   8, 8, reuse, is_training=is_training) # 8
    #l22= upconvlayer(3,     l2,    3,   128,         64,    8, reuse, is_training=is_training) # 8
    #l3 = upconvlayer(4,     l22,   3,   64,          32,   16, reuse, is_training=is_training) # 8->16
    #l4 = upconvlayer(5,     l3,    3,   32,          16,   16, reuse, is_training=is_training)
    #l5 = upconvlayer(6,     l4,    3,   16,          16,   32, reuse, is_training=is_training)
    #recons_logits = upconvlayer(7, l5,3,16,          1,    32, reuse, is_training=is_training, nonlin=tf.identity, dobn=False)

    l1 = upconvlayer(1,     codes, 1,   num_latents, 512,   1, reuse, is_training=is_training) 
    l2 = upconvlayer_tr(2,  l1,    8,   512,         128,   8, 8, reuse, is_training=is_training) # 8
    l22= upconvlayer(3,     l2,    3,   128,         64,    8, reuse, is_training=is_training) # 8
    l3 = upconvlayer(4,     l22,   3,   64,          64,   16, reuse, is_training=is_training) # 8->16
    l4 = upconvlayer(5,     l3,    3,   64,          32,   16, reuse, is_training=is_training)
    l5 = upconvlayer(6,     l4,    3,   32,          32,   32, reuse, is_training=is_training)
    l55= upconvlayer(7,     l5,    3,   32,          16,   32, reuse, is_training=is_training)
    recons_logits = upconvlayer(8,l55,3,16,          1,    32, reuse, is_training=is_training, nonlin=tf.identity, dobn=False)
    recons = tf.sigmoid(recons_logits)
    return recons, recons_logits

def classifier(codes, f_params_dim, is_training, reuse=False):
    print('classifier')
    batch_size = codes.get_shape().as_list()[0]
    codes = tf.reshape(codes, [batch_size, 1, -1, f_params_dim])
    inv_codes_mat = tf.matmul(codes, tf.transpose(codes, [0, 1, 3, 2]))
    inv_codes_mat = el.get_utr(inv_codes_mat)
    #inv_codes_mat = tf.reshape(inv_codes_mat, [batch_size, -1])
    feats = inv_codes_mat
    inpdim = feats.get_shape().as_list()[1]
    print(feats)

    def mul_noise(inp, is_training, stddev=0.5):
        switch = tf.cast(is_training, tf.float32)
        noise = 1.0 + switch*tf.truncated_normal(inp.get_shape(), stddev=stddev)
        out = inp*noise
        return out

    def add_noise(inp, is_training, stddev=0.5):
        switch = tf.cast(is_training, tf.float32)
        noise = tf.truncated_normal(inp.get_shape(), stddev=stddev)
        out = inp + switch*noise
        return out

    ##feats = add_noise(feats, is_training)
    ##feats = mul_noise(feats, is_training)
    #l1 = mlplayer(0, feats, inpdim, 256, reuse=reuse, is_training=is_training, dobn=True)
    ##l1 = add_noise(l1, is_training)
    ##l1 = mul_noise(l1, is_training)
    #y_logits = mlplayer(1, l1, 256, 10, nonlin=tf.identity, reuse=reuse, is_training=is_training, dobn=False)

    ## TODO retraining 10_2
    feats = add_noise(feats, is_training, 0.1)
    #feats = mul_noise(feats, is_training)
    l1 = mlplayer(0, feats, inpdim, 256, reuse=reuse, is_training=is_training, dobn=True)
    l1 = add_noise(l1, is_training, 0.1)
    l2 = mlplayer(1, l1, 256, 128, reuse=reuse, is_training=is_training, dobn=True)
    l2 = add_noise(l2, is_training, 0.1)
    y_logits = mlplayer(2, l2, 128, 10, nonlin=tf.identity, reuse=reuse, is_training=is_training, dobn=False)

    ### TODO retraining 10_2m
    #feats = mul_noise(feats, is_training)
    ##feats = mul_noise(feats, is_training)
    #l1 = mlplayer(0, feats, inpdim, 256, reuse=reuse, is_training=is_training, dobn=True)
    #l1 = mul_noise(l1, is_training)
    #l2 = mlplayer(1, l1, 256, 128, reuse=reuse, is_training=is_training, dobn=True)
    #y_logits = mlplayer(2, l2, 128, 10, nonlin=tf.identity, reuse=reuse, is_training=is_training, dobn=False)

    return y_logits


def classifier_loss(y_true, y_logits, class_weight):
    #y_logits = y_logits/(0.001 + class_weight)
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_logits)

def bernoulli_xentropy(target, output):
    """Cross-entropy for Bernoulli variables"""
    target = 3*target-1
    output = 0.8999*output + 0.1000
    wx_entropy = -(98.0*target*tf.log(output) + 2.0*(1. - target)*tf.log(1.0 - output))/100.0
    return tf.reduce_sum(wx_entropy, axis=(1,2,3,4))


def identity_stl_transmats(batch_size):
    _, stl_transmat_inp, _, _ = random_transmats(batch_size)
    stl_transmat_inp[:,:,:] = 0.0
    stl_transmat_inp[:,0,0] = 1.0
    stl_transmat_inp[:,1,1] = 1.0
    stl_transmat_inp[:,2,2] = 1.0
    return stl_transmat_inp.astype(np.float32)


def random_transmats(batch_size):
    """ Random rotations in 3D
    """
    min_scale = 1.0
    max_scale = 1.0

    aug_scale_step = 1./32
    aug_shift_step = 1./32
    aug_max_scale = 2.500001
    aug_max_shift = 2.500001

    #params_aug_scale = aug_shift_step*np.random.random_integers(-aug_max_scale, aug_max_scale, size=(batch_size, 3))
    params_aug_scale = aug_scale_step*aug_max_scale*2*(np.random.rand(batch_size, 3)-1)
    params_aug_scale[:,:] += 1.0
    #params_aug_scale[:,[1,2]] = 1.0
    #aug_3dshift = aug_shift_step*np.random.random_integers(-aug_max_shift, aug_max_shift, size=(batch_size, 3, 1))
    aug_3dshift = aug_shift_step*aug_max_shift*2*(np.random.rand(batch_size, 3, 1)-1)
    aug_3dscalemat = get_3dscalemat(params_aug_scale)
    stl_transmat_aug = np.concatenate([aug_3dscalemat, aug_3dshift], axis=2)

    if True:
        params_inp_rot = np.pi*2*(np.random.rand(batch_size, 3)-0.5)
        params_inp_rot[:,[1,2]] = 0.0
        params_inp_scale = 1.0 + 0.0*np.random.rand(batch_size, 3)

        params_trg_rot = np.pi*2*(np.random.rand(batch_size, 3)-0.5)
        params_trg_rot[:,[1,2]] = 0.0
        params_trg_scale = 1.0 + 0.0*np.random.rand(batch_size, 3)
    else:
        params_inp_rot = np.pi*2*(np.random.rand(batch_size, 3)-0.5)
        #params_inp_rot[:,1] = params_inp_rot[:,1]/2
        params_inp_scale = min_scale + (max_scale-min_scale)*np.random.rand(batch_size, 3)

        params_trg_rot = np.pi*2*(np.random.rand(batch_size, 3)-0.5)
        #params_trg_rot[:,1] = params_trg_rot[:,1]/2
        params_trg_scale = min_scale + (max_scale-min_scale)*np.random.rand(batch_size, 3)


    inp_3drotmat = get_3drotmat(params_inp_rot)
    inp_3dscalemat = get_3dscalemat(params_inp_scale)

    trg_3drotmat = get_3drotmat(params_trg_rot)
    trg_3dscalemat = get_3dscalemat(params_trg_scale)

    # scale and rot because inverse warp in stl
    stl_transmat_inp = np.matmul(inp_3dscalemat, inp_3drotmat)
    stl_transmat_trg = np.matmul(trg_3dscalemat, trg_3drotmat)

    f_params_inp = np.zeros([batch_size, 2, 2])
    # was like this:
    #cur_rotmat = np.matmul(trg_3drotmat, inp_3drotmat.transpose([0,2,1]))
    cur_rotmat = np.matmul(trg_3drotmat.transpose([0,2,1]), inp_3drotmat)
    f_params_inp = set_f_params_rot(f_params_inp, cur_rotmat)
    
    #f_params_inp = np.zeros([batch_size, 3, 3])
    ## was like this:
    ##cur_rotmat = np.matmul(trg_3drotmat, inp_3drotmat.transpose([0,2,1]))
    #cur_rotmat = np.matmul(trg_3drotmat.transpose([0,2,1]), inp_3drotmat)
    #f_params_inp = set_f_params_rot(f_params_inp, cur_rotmat)

    # TODO
    #for i in xrange(3):
    #    inp_f_2dscalemat = get_2drotscalemat(params_inp_scale[:, i], min_scale, max_scale)
    #    trg_f_2dscalemat = get_2drotscalemat(params_trg_scale[:, i], min_scale, max_scale)
    #    cur_f_scalemat = np.matmul(trg_f_2dscalemat, inp_f_2dscalemat.transpose([0,2,1]))
    #    f_params_inp = set_f_params_scale(f_params_inp, i, cur_f_scalemat)

    stl_transmat_aug = stl_transmat_aug.astype(np.float32)
    stl_transmat_inp = stl_transmat_inp.astype(np.float32)
    stl_transmat_trg = stl_transmat_trg.astype(np.float32)
    f_params_inp = f_params_inp.astype(np.float32)
    return stl_transmat_aug, stl_transmat_inp, stl_transmat_trg, f_params_inp

def set_f_params_rot(f_params, rotmat):
    f_params[:,0:2,0:2] = rotmat[:,1:3,1:3]
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


############################################

def load_sess(sess, load_path, load_classifier=True):
    ckpt = tf.train.get_checkpoint_state(load_path)
    print('loading from', load_path)
    
    if ckpt and ckpt.model_checkpoint_path:
        vars_to_restore = tf.global_variables()
        res_saver = tf.train.Saver(vars_to_restore)
        if not load_classifier:
            vars_to_pop = [var for var in vars_to_restore if 'classifier' in var.name]
            for var in vars_to_pop:
                vars_to_restore.remove(var)
        # Restores from checkpoint
        model_checkpoint_path = os.path.abspath(ckpt.model_checkpoint_path)
        print(model_checkpoint_path)
        res_saver.restore(sess, model_checkpoint_path)
        
        # Restores from checkpoint
        model_checkpoint_path = os.path.abspath(ckpt.model_checkpoint_path)
        print(model_checkpoint_path)
        res_saver.restore(sess, model_checkpoint_path)
    else:
        print('No checkpoint file found')
        return None
    return sess

def vis(inputs, outputs, ops, opt, data):
    print('in vis')
    """Training loop"""
    # Unpack inputs, outputs and ops
    x, global_step, stl_params_aug, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training, y_true, test_y_true = inputs
    loss, merged, test_recon, x_in, test_x_in = outputs
    train_op = ops
    
    # For checkpoints
    gs = 0
    start = time.time()
    
    saver = tf.train.Saver()
    sess = tf.Session()

    # Initialize variables
    if len(opt['load_path'])>0:
        sess = load_sess(sess, opt['load_path'], False)
    else:
        print('No model to load')
        return
    
    # check test set
    num_steps = 50#data.test.num_steps(1)
    for step_i in xrange(num_steps):
        print(step_i)
        val_recons = []
        val_vox = []
        #pick a random initial transformation
        _, cur_stl_params_in, _, cur_f_params = random_transmats(1)

        max_angles = 5
        
        cur_x, cur_y_true = data.test.next_batch(1)
        fangles = np.linspace(-np.pi, np.pi, num=max_angles*max_angles)
        rot_ax = 0

        for i in xrange(max_angles):
            for j in xrange(max_angles):
                cur_f_params_j = update_f_params(cur_f_params, 0, rot_ax, fangles[i*max_angles + j])
                feed_dict = {
                            test_x : cur_x,
                            test_stl_params_in : cur_stl_params_in, 
                            val_f_params: cur_f_params_j,
                            is_training : False
                        }

                y = sess.run(test_recon, feed_dict=feed_dict)
                val_recons.append(y[0,:,:,:,:].copy())
                val_vox.append(y[0,:,:,:,0].copy()>0.5)

                if j==0:
                    cur_x_in = sess.run(test_x_in, feed_dict=feed_dict)
                    val_vox_in = cur_x_in[0,:,:,:,0].copy()>0.5
        
        samples_ = np.stack(val_recons)
        print(samples_.shape)

        tile_image = np.reshape(samples_, [max_angles*max_angles, opt['outsize'][0], opt['outsize'][1], opt['outsize'][2], opt['color_chn']])
        tile_image_d, tile_image_h, tile_image_w = get_imgs_from_vol(tile_image, max_angles, max_angles)

        save_folder = './vis/' + opt['flag'] + '/'
        checkFolder(save_folder)
        save_name = save_folder + '/image_%04d' % step_i
        imsave(save_name + '_d.png', tile_image_d) 
        imsave(save_name + '_h.png', tile_image_h) 
        imsave(save_name + '_w.png', tile_image_w) 

        for i, v in enumerate(val_vox):
            save_name = save_folder + '/binvox_%04d_%004d' % (step_i, i)
            save_binvox(v, save_name)
        save_name = save_folder + '/inp_binvox_%04d' % step_i
        save_binvox(val_vox_in, save_name)

        tile_image_d, tile_image_h, tile_image_w = get_imgs_from_vol(cur_x_in, 1, 1)
        save_name = save_folder + '/input_%04d' % step_i
        imsave(save_name + '_d.png', tile_image_d) 
        imsave(save_name + '_h.png', tile_image_h) 
        imsave(save_name + '_w.png', tile_image_w) 



def test(inputs, outputs, ops, opt, data):
    """Training loop"""
    # Unpack inputs, outputs and ops
    x, global_step, stl_params_aug, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training, y_true, test_y_true = inputs
    loss, merged, test_recon, recons, rec_loss, c_loss, y_logits, test_y_logits, y_pred, test_y_pred = outputs
    train_op = ops
    
    # For checkpoints
    gs = 0
    start = time.time()
    
    saver = tf.train.Saver()
    sess = tf.Session()

    # Initialize variables
    if len(opt['load_path'])>0:
        sess = load_sess(sess, opt['load_path'], True)
    else:
        print('No model to load')
        return
    
    # check test set
    num_steps = data.test.num_steps(1)
    test_acc = 0
    #cur_stl_params_in = identity_stl_transmats(1)
    _, cur_stl_params_in, _, cur_f_params = random_transmats(1)
    for step_i in xrange(num_steps):
        cur_x, cur_y_true = data.test.next_batch(1)
        feed_dict = {
                    test_x: cur_x,
                    test_stl_params_in: cur_stl_params_in, 
                    is_training : False
                }
        cur_y_pred = sess.run(test_y_pred, feed_dict=feed_dict)
        cur_test_y = (np.argmax(cur_y_pred, axis=1))
        cur_test_y_pred = (np.argmax(cur_y_true, axis=1))
        #if step_i==0:
        print((cur_test_y, cur_test_y_pred))

        diff = (cur_test_y - cur_test_y_pred)==0
        diff = diff.astype(np.float32)
        test_acc += np.sum(diff)

    print('correctly classified:', test_acc)
    print('num_steps', num_steps)

    

def train(inputs, outputs, ops, opt, data):
    """Training loop"""
    # Unpack inputs, outputs and ops
    x, global_step, stl_params_aug, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training, y_true, test_y_true = inputs
    loss, merged, test_recon, recons, rec_loss, c_loss, y_logits, test_y_logits, y_pred, test_y_pred = outputs
    train_op = ops
    
    # For checkpoints
    gs = 0
    start = time.time()
    
    saver = tf.train.Saver()
    sess = tf.Session()

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    if len(opt['load_path'])>0:
        sess = load_sess(sess, opt['load_path'], True)
    else:
        print('Training from scratch')
    
    train_writer = tf.summary.FileWriter(opt['summary_path'], sess.graph)

    # Training loop
    for epoch in xrange(opt['n_epochs']):
        # Learning rate
        exponent = sum([epoch > i for i in opt['lr_schedule']])
        current_lr = opt['lr']*np.power(0.1, exponent)
        

        #if opt['do_classify'] and epoch%1==0:
        #    # check validation accuracy
        #    num_steps = data.validation.num_steps(opt['mb_size'])-1
        #    val_acc = 0
        #    for step_i in xrange(num_steps):
        #        cur_stl_params_aug, cur_stl_params_in, _, _ = random_transmats(opt['mb_size'])
        #        cur_x, cur_y_true = data.validation.next_batch(opt['mb_size'])
        #        feed_dict = {
        #                    x: cur_x,
        #                    y_true: cur_y_true,
        #                    stl_params_aug: cur_stl_params_aug, 
        #                    stl_params_in: cur_stl_params_in, 
        #                    is_training : False
        #                }
        #        cur_y_pred = sess.run(y_pred, feed_dict=feed_dict)
        #        val_y = (np.argmax(cur_y_pred, axis=1))
        #        val_y_pred = (np.argmax(cur_y_true, axis=1))
        #        if step_i==0:
        #            print(val_y)
        #            print(val_y_pred)

        #        diff = (val_y - val_y_pred)==0
        #        diff = diff.astype(np.float32)
        #        val_acc += np.sum(diff)/opt['mb_size']

        #    val_acc /= num_steps
        #    print('validation accuracy', val_acc)
        
        # Train
        train_loss = 0.
        train_rec_loss = 0.
        train_c_loss = 0.
        train_acc = 0.
        train_acc10 = 0.

        # Run training steps
        num_steps = data.train.num_steps(opt['mb_size'])
        for step_i in xrange(num_steps):
            cur_stl_params_aug, cur_stl_params_in, cur_stl_params_trg, cur_f_params = random_transmats(opt['mb_size'])
            ops = [global_step, loss, merged, train_op, rec_loss]
            cur_x, cur_y_true = data.train.next_batch(opt['mb_size'])
            
            feed_dict = {
                        x: cur_x,
                        y_true: cur_y_true,
                        stl_params_aug: cur_stl_params_aug,
                        stl_params_in: cur_stl_params_in, 
                        stl_params_trg: cur_stl_params_trg, 
                        f_params : cur_f_params, 
                        lr : current_lr,
                        is_training : True
                    }

            gs, l, summary, __, rec_l = sess.run(ops, feed_dict=feed_dict)
            train_loss += l
            train_rec_loss += rec_l
            if opt['do_classify']:
                c_ops = [c_loss, y_pred]
                c_l, cur_y_pred = sess.run(c_ops, feed_dict=feed_dict)
                train_c_loss += c_l
                diff = (cur_y_pred*cur_y_true)
                diff = diff.astype(np.float32)
                cur_acc = np.sum(diff)/opt['mb_size']
                train_acc += cur_acc
                train_acc10 += cur_acc

                if step_i % 10==9:
                    train_acc10 /= 10
                    print('[{:03f}]: train_loss: {:03f}. rec_loss: {:03f}. c_loss: {:03f}. train_acc: {:03f}.'.format(float(step_i)/num_steps, l, rec_l, c_l, train_acc10))

                    train_acc10 = 0.
            else:
                if step_i % 10==0:
                    print('[{:03f}]: train_loss: {:03f}.'.format(float(step_i)/num_steps, l))

            assert not np.isnan(l), 'Model diverged with loss = NaN'

            if step_i % 10==0:
                # Summary writers
                train_writer.add_summary(summary, gs)

        train_loss /= num_steps
        train_rec_loss /= num_steps
        train_c_loss /= num_steps
        train_acc /= num_steps

        print('[{:03d}]: train_loss: {:03f}. rec_loss: {:03f}. c_loss: {:03f}. train_acc: {:03f}.'.format(epoch, train_loss, train_rec_loss, train_c_loss, train_acc))

        # Save model
        if epoch % FLAGS.save_step == 0 or epoch+1==opt['n_epochs']:
            path = saver.save(sess, opt['save_path'] + 'model.ckpt', epoch)
            print('Saved model to ' + path)
        

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
    
    opt['mb_size'] = 16
    opt['n_epochs'] = 2000
    opt['lr_schedule'] = [1900]
    opt['lr'] = 1e-4

    opt['vol_size'] = [32,32,32]
    pad_size = 0#int(np.ceil(np.sqrt(3)*opt['vol_size'][0]/2)-opt['vol_size'][0]/2)
    opt['outsize'] = [i + 2*pad_size for i in opt['vol_size']]
    stl = AffineVolumeTransformer(opt['outsize'])
    opt['color_chn'] = 1
    opt['f_params_dim'] = 2# + 2*3 # rotation matrix is 3x3 and we have 3 axis scalings implemented as 2x2 rotations
    opt['num_latents'] = opt['f_params_dim']*100
    # TODO
    opt['stl_size'] = 3 # no translation

    #opt['flag'] = 'modelnet_classify100_cont2'
    #opt['flag'] = 'modelnet_classify1000_unbalanced'
    #opt['flag'] = 'modelnet_classify100_ae'
    #opt['flag'] = 'modelnet_classify1000_scratch'
    #opt['flag'] = 'modelnet_classify10000_cont2'
    #opt['flag'] = 'modelnet_vis'
    #opt['flag'] = 'modelnet_classify100_cont'
    #opt['flag'] = 'modelnet_classify1000_scratch'
    #opt['flag'] = 'modelnet_classify10000_scratch'
    #opt['flag'] = 'modelnet2_classify1000'
    #opt['flag'] = 'modelnet2_classify100_ta' # threshold augmentation
    #opt['flag'] = 'modelnet2_classify10_2' # threshold augmentation, 2-layer mlp 256-128 with additive noise
    #opt['flag'] = 'modelnet2_classify1_2' # threshold augmentation, 2-layer mlp 256-128 with additive noise of 0.1
    #opt['flag'] = 'modelnet2_classify10_2m' # threshold augmentation, 2-layer mlp 256-128 with multiplicative noise
    #opt['flag'] = 'modelnet2_classify1_ta' # threshold augmentation
    #opt['flag'] = 'modelnet2_classify100'
    opt['flag'] = 'modelnet2_test'
    opt['summary_path'] = dir_ + '/summaries/autotrain_{:s}'.format(opt['flag'])
    opt['save_path'] = dir_ + '/checkpoints/autotrain_{:s}/'.format(opt['flag'])
    
    ###
    #opt['load_path'] = ''
    opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet2_classify1_2/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet2_classify10_2/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet2_classify10_2m/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet2/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet2_classify1_ta/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet2_classify100_ta/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet2_classify1000/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet_cont/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet_classify100_cont/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet_classify1000_cont/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet_classify10000_cont/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet_cont/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet_classify100_scratch/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet_classify1000_cont/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_modelnet_classify1000_scratch/'
    opt['do_classify'] = True
    
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
    y_true = tf.placeholder(tf.int32, [opt['mb_size'],10], name='y_true')

    stl_params_aug = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],1+opt['stl_size']], name='stl_params_aug')
    stl_params_in  = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],opt['stl_size']], name='stl_params_in')
    stl_params_trg = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],opt['stl_size']], name='stl_params_trg')
    f_params = tf.placeholder(tf.float32, [opt['mb_size'], opt['f_params_dim'], opt['f_params_dim']], name='f_params')

    test_x = tf.placeholder(tf.float32, [1,opt['vol_size'][0],opt['vol_size'][1],opt['vol_size'][2],opt['color_chn']], name='test_x')
    test_y_true = tf.placeholder(tf.int32, [1,10], name='test_y_true')
    test_stl_params_in  = tf.placeholder(tf.float32, [1,opt['stl_size'],opt['stl_size']], name='test_stl_params_in')
    val_f_params = tf.placeholder(tf.float32, [1, opt['f_params_dim'], opt['f_params_dim']], name='val_f_params') 
    paddings = tf.convert_to_tensor(np.array([[0,0], [pad_size,pad_size], [pad_size,pad_size], [pad_size, pad_size], [0,0]]), dtype=tf.int32)
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.placeholder(tf.float32, [], name='lr')
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    
    # augment training volume
    flip_dim = tf.random_uniform([1], minval=1, maxval=3, dtype=tf.int32) # maxval is not inclusive
    x_flip = tf.reverse(x, axis=flip_dim) # can flip axis = 1 or 2
    x_aug = spatial_transform(stl, x_flip, stl_params_aug, paddings, threshold=True)
    
    # generate input and target volumes
    x_in = spatial_transform(stl, x_aug, stl_params_in)
    x_trg = spatial_transform(stl, x_aug, stl_params_trg)

    # Build the training model
    recons, codes, recons_logits = autoencoder(x_in, opt['num_latents'], f_params, is_training)
    
    # Test model
    test_x_pad = spatial_pad(test_x, paddings)
    test_x_in = spatial_transform(stl, test_x_pad, test_stl_params_in)
    #test_x_in = test_x_pad
    test_recon, test_codes, _ = autoencoder(test_x_in, opt['num_latents'], val_f_params, is_training, reuse=True)

    # LOSS
    rec_loss = tf.reduce_mean(bernoulli_xentropy(x_trg, recons))
    y_logits = None
    y_pred = None
    test_y_logits = None
    test_y_pred = None
    c_loss = 0

    if opt['do_classify']:
        y_logits = classifier(codes, opt['f_params_dim'], is_training) 
        y_pred = tf.nn.softmax(y_logits)
        test_y_logits = classifier(test_codes, opt['f_params_dim'], is_training, reuse=True)
        test_y_pred = tf.nn.softmax(test_y_logits)
        c_loss = classifier_loss(y_true, y_logits, data.train.class_balance)
        c_loss = 1.*tf.reduce_mean(c_loss)
    loss = rec_loss + c_loss
    
    # Summaries
    tf_vol_summary('recons', recons) 
    tf_vol_summary('inputs', x_in) 
    tf_vol_summary('targets', x_trg) 
    tf.summary.scalar('Rec Loss', rec_loss)
    tf.summary.scalar('C Loss', c_loss)
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('LearningRate', lr)
    merged = tf.summary.merge_all()
    
    # Build optimizer
    optim = tf.train.AdamOptimizer(lr)
    #optim = tf.train.MomentumOptimizer(lr, momentum=0.1, use_nesterov=True)
    train_op = optim.minimize(loss, global_step=global_step)
    
    # Set inputs, outputs, and training ops
    inputs = [x, global_step, stl_params_aug, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training, y_true, test_y_true]
    ops = [train_op]
    
    print(FLAGS.VIS)
    if FLAGS.VIS:
        outputs = [loss, merged, test_recon, x_in, test_x_in]
        return vis(inputs, outputs, ops, opt, data)

    outputs = [loss, merged, test_recon, recons, rec_loss, c_loss, y_logits, test_y_logits, y_pred, test_y_pred]

    print(FLAGS.TEST)
    if FLAGS.TEST:
        return test(inputs, outputs, ops, opt, data)
    # Train
    return train(inputs, outputs, ops, opt, data)


if __name__ == '__main__':
    tf.app.run()
