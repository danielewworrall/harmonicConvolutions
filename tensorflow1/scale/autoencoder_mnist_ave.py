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
import mnist_loader
import equivariant_loss as el
from spatial_transformer import transformer

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
flags.DEFINE_boolean('DaniyarSleepy', True, 'Sleepy execution environment')
flags.DEFINE_boolean('Mac', False, 'Mac execution environment')
flags.DEFINE_boolean('VIS', False, 'Visualize feature space transformations')

##---------------------

################ DATA #################

def load_data():
    mnist = mnist_loader.read_data_sets("/tmp/data/", one_hot=True)
    return mnist


def checkFolder(dir):
    """Checks if a folder exists and creates it if not.
    dir: directory
    Returns nothing
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


############## MODEL ####################

def spatial_transform(x, t_params, imsh):
    """Spatial transformer wtih shapes sets"""
    xsh = x.get_shape()
    #x_in = transformer(x, t_params, imsh)
    batch_size = t_params.get_shape().as_list()[0]

    if t_params.get_shape().as_list()[2]==2:
        shiftmat = tf.zeros([batch_size,2,1], dtype=tf.float32)
        t_params = tf.concat([t_params, shiftmat], axis=2)
        
    t_params = tf.reshape(t_params, [-1, 6])
    x_in = transformer(x, t_params, imsh)
    x_in.set_shape(xsh)
    return x_in

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
    strides = [1, stride, stride, 1]
    with tf.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = variable(scopename + '_kernel', [ksize, ksize, inpdim, outdim])
        bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
        linout = tf.nn.conv2d(inp, kernel, strides=strides, padding=padding)
        linout = tf.nn.bias_add(linout, bias)
        if dobn:
            bnout = bn4d(linout, is_training, reuse=reuse)
        else:
            bnout = linout
        out = nonlin(bnout, name=scopename + '_nonlin')
    print(' out:', out)
    return out

def bn4d(X, train_phase, decay=0.99, name='batchNorm', reuse=False):
    assert len(X.get_shape().as_list())==4, 'input to bn4d must be 4d tensor'

    n_out = X.get_shape().as_list()[-1]
    
    beta = tf.get_variable('beta_'+name, dtype=tf.float32, shape=n_out, initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma_'+name, dtype=tf.float32, shape=n_out, initializer=tf.constant_initializer(1.0))
    pop_mean = tf.get_variable('pop_mean_'+name, dtype=tf.float32, shape=n_out, trainable=False)
    pop_var = tf.get_variable('pop_var_'+name, dtype=tf.float32, shape=n_out, trainable=False)

    batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2], name='moments_'+name)
    
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




def upconvlayer(i, inp, ksize, inpdim, outdim, outshape, reuse, nonlin=tf.nn.elu, dobn=True, is_training=True):
    scopename = 'upconv_layer' + str(i)
    inpshape = inp.get_shape().as_list()[-2]
    print(scopename)
    print(' input:', inp)
    output_shape = [outshape, outshape]
    padding='SAME'
    with tf.variable_scope(scopename) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = variable(scopename + '_kernel', [ksize, ksize, inpdim, outdim])
        bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
        if outshape>inpshape:
            inp_resized = tf.image.resize_nearest_neighbor(inp, output_shape)
        else:
            inp_resized = inp
        linout = tf.nn.conv2d(inp_resized, kernel, strides=[1, 1, 1, 1], padding=padding)
        linout = tf.nn.bias_add(linout, bias)
        if dobn:
            bnout = bn4d(linout, is_training, reuse=reuse)
        else:
            bnout = linout
        out = nonlin(bnout, name=scopename + 'nonlin')
    print(' out:', out)
    return out

def encoder_conv(x, num_latents, is_training, reuse=False):
    """Encoder Conv"""
    l0 = convlayer(0, x,  3,   1,    8, 1, reuse, padding='VALID', nonlin=tf_nn_lrelu, is_training=is_training) # 26
    l1 = convlayer(1, l0, 4,  8,    16, 2, reuse, padding='VALID', nonlin=tf_nn_lrelu, is_training=is_training) # 13
    l2 = convlayer(2, l1, 4,  16,   32, 2, reuse, padding='VALID', nonlin=tf_nn_lrelu, is_training=is_training) # 5
    l3 = convlayer(3, l2, 3,  32,   64, 1, reuse, padding='VALID', nonlin=tf_nn_lrelu, is_training=is_training) # 3
    l4 = convlayer(4, l3, 3,  64,  256, 1, reuse, padding='VALID', nonlin=tf_nn_lrelu, is_training=is_training) # 1
    l5 = convlayer(5, l4, 1,  256,  num_latents, 1, reuse, nonlin=tf.identity, is_training=is_training, padding='VALID', dobn=False)
    l6 = tf.reshape(l5, [-1, num_latents])
    return l6


def decoder_conv(z, is_training, reuse=False):
    """Decoder Conv"""
    batch_size = z.get_shape().as_list()[0]
    num_latents = z.get_shape().as_list()[-1]
    z = tf.reshape(z, [batch_size, 1, 1, num_latents])
    l5 = upconvlayer(5, z,  1,  num_latents, 256,   1, reuse, nonlin=tf_nn_lrelu, is_training=is_training)
    l4 = upconvlayer(4, l5, 3,  256, 64,   3, reuse, nonlin=tf_nn_lrelu, is_training=is_training)
    l3 = upconvlayer(3, l4, 3,  64,  32,   5, reuse, nonlin=tf_nn_lrelu, is_training=is_training)
    l2 = upconvlayer(2, l3, 4,  32,  16,  13, reuse, nonlin=tf_nn_lrelu, is_training=is_training)
    l1 = upconvlayer(1, l2, 4,  16,   8,  26, reuse, nonlin=tf_nn_lrelu, is_training=is_training)
    l0 = upconvlayer(0, l1, 3,  8,    1,  28, reuse, nonlin=tf.identity, is_training=is_training, dobn=False)

    recons_logits = l0
    recons = tf.sigmoid(recons_logits)
    return recons, recons_logits

def autoencoder_conv(x, x2, num_latents, f_params, f_params2, is_training, reuse=False):
    x = tf.reshape(x, [-1, 28, 28, 1])
    x2 = tf.reshape(x2, [-1, 28, 28, 1])
    """Build a model to rotate features"""
    with tf.variable_scope('mainModel', reuse=reuse) as scope:
        with tf.variable_scope("encoder_conv", reuse=reuse) as scope:
            codes = encoder_conv(x, num_latents, is_training, reuse=reuse)
        with tf.variable_scope("encoder_conv", reuse=True) as scope:
            codes2 = encoder_conv(x2, num_latents, is_training, reuse=True)

        with tf.variable_scope("feature_transformer", reuse=reuse) as scope:
            code_shape = codes.get_shape()
            batch_size = code_shape.as_list()[0]
            codes = tf.reshape(codes, [batch_size, -1])
            codes_transformed = el.feature_transform_matrix_n(codes, codes.get_shape(), f_params)
            codes_transformed = tf.reshape(codes_transformed, code_shape)

        with tf.variable_scope("feature_transformer", reuse=True) as scope:
            codes2 = tf.reshape(codes2, [batch_size, -1])
            codes2_transformed = el.feature_transform_matrix_n(codes2, codes.get_shape(), f_params2)
            codes2_transformed = tf.reshape(codes2_transformed, code_shape)

            coeff = tf.random_uniform([1], minval=0.0, maxval=1.0)[0]
            codes_ave = coeff*codes_transformed + (1.-coeff)*codes2_transformed

        with tf.variable_scope("decoder_conv", reuse=reuse) as scope:
            recons, recons_logits = decoder_conv(codes_ave, is_training, reuse=reuse)

    recons = tf.reshape(recons, [-1, 28*28])
    recons_logits = tf.reshape(recons_logits, [-1, 28*28])
    return recons, codes, codes_transformed, codes2_transformed, recons_logits, codes_ave

def encoder(x, num_latents, is_training, reuse=False):
    """Encoder MLP"""
    l1 = mlplayer(0, x, 784, 512, is_training=is_training, reuse=reuse, dobn=True)
    l2 = mlplayer(1, l1, 512, 512, is_training=is_training, reuse=reuse, dobn=True)
    codes = mlplayer(2, l2, 512, num_latents, is_training=is_training, reuse=reuse, dobn=False, nonlin=tf.identity)
    return codes


def decoder(z, is_training, reuse=False):
    """Encoder MLP"""
    l2 = mlplayer(3, z, z.get_shape()[1], 512, is_training=is_training, reuse=reuse, dobn=True)
    l1 = mlplayer(4, l2, 512, 512, is_training=is_training, reuse=reuse, dobn=True)
    recons_logits = mlplayer(5, l1, 512, 784, is_training=is_training, reuse=reuse, dobn=False, nonlin=tf.identity)
    recons = tf.sigmoid(recons_logits)
    return recons, recons_logits

def autoencoder(x, x2, num_latents, f_params, f_params2, is_training, reuse=False):
    """Build a model to rotate features"""
    with tf.variable_scope('mainModel', reuse=reuse) as scope:
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            codes = encoder(x, num_latents, is_training, reuse=reuse)
        with tf.variable_scope("encoder", reuse=True) as scope:
            codes2 = encoder(x2, num_latents, is_training, reuse=True)

        with tf.variable_scope("feature_transformer", reuse=reuse) as scope:
            code_shape = codes.get_shape()
            batch_size = code_shape.as_list()[0]
            codes = tf.reshape(codes, [batch_size, -1])
            codes_transformed = el.feature_transform_matrix_n(codes, codes.get_shape(), f_params)
            codes_transformed = tf.reshape(codes_transformed, code_shape)

        with tf.variable_scope("feature_transformer", reuse=True) as scope:
            codes2 = tf.reshape(codes2, [batch_size, -1])
            codes2_transformed = el.feature_transform_matrix_n(codes2, codes.get_shape(), f_params2)
            codes2_transformed = tf.reshape(codes2_transformed, code_shape)

            coeff = tf.random_uniform([1], minval=0.0, maxval=1.0)[0]
            codes_ave = coeff*codes_transformed + (1.-coeff)*codes2_transformed

        with tf.variable_scope("decoder", reuse=reuse) as scope:
            recons, recons_logits = decoder(codes_ave, is_training, reuse=reuse)

    return recons, codes, codes_transformed, codes2_transformed, recons_logits, codes_ave


def bernoulli_xentropy(x, recon_test):
    """Cross-entropy for Bernoulli variables"""
    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_test)
    return tf.reduce_mean(tf.reduce_sum(x_entropy, axis=(1)))

def random_transmats2(batch_size):
    """ Random rotations in 2D
    """

    if True:
        params_inp_rot = np.pi*2*(np.random.rand(batch_size)-0.5)
        params_inp_rot2 = np.pi*2*(np.random.rand(batch_size)-0.5)
        params_trg_rot = np.pi*2*(np.random.rand(batch_size)-0.5)

    inp_2drotmat = get_2drotmat(params_inp_rot)
    inp_2drotmat2 = get_2drotmat(params_inp_rot2)
    trg_2drotmat = get_2drotmat(params_trg_rot)

    f_params_inp = np.matmul(trg_2drotmat.transpose([0,2,1]), inp_2drotmat)
    f_params_inp2 = np.matmul(trg_2drotmat.transpose([0,2,1]), inp_2drotmat2)
    
    stl_transmat_inp = inp_2drotmat.astype(np.float32)
    stl_transmat_inp2 = inp_2drotmat2.astype(np.float32)
    stl_transmat_trg = trg_2drotmat.astype(np.float32)
    f_params_inp = f_params_inp.astype(np.float32)
    f_params_inp2 = f_params_inp2.astype(np.float32)

    return stl_transmat_inp, stl_transmat_inp2, stl_transmat_trg, f_params_inp, f_params_inp2

def random_transmats(batch_size):
    """ Random rotations in 2D
    """

    if True:
        params_inp_rot = np.pi*2*(np.random.rand(batch_size)-0.5)
        params_trg_rot = np.pi*2*(np.random.rand(batch_size)-0.5)

    inp_2drotmat = get_2drotmat(params_inp_rot)
    trg_2drotmat = get_2drotmat(params_trg_rot)

    f_params_inp = np.matmul(trg_2drotmat.transpose([0,2,1]), inp_2drotmat)
    
    stl_transmat_inp = inp_2drotmat.astype(np.float32)
    stl_transmat_trg = trg_2drotmat.astype(np.float32)
    f_params_inp = f_params_inp.astype(np.float32)

    return None, stl_transmat_inp, stl_transmat_trg, f_params_inp

def update_f_params(f_params, theta):
    # do 2x2 rotation
    angles = np.zeros([1])
    angles[:] = theta
    inp_2drotmat = get_2drotmat(angles)
    f_params = inp_2drotmat
    return f_params

############################################

def load_sess(sess, load_path):
    ckpt = tf.train.get_checkpoint_state(load_path)
    print('loading from', load_path)
    
    if ckpt and ckpt.model_checkpoint_path:
        vars_to_restore = tf.global_variables()
        res_saver = tf.train.Saver(vars_to_restore)

        # Restores from checkpoint
        model_checkpoint_path = load_path + ckpt.model_checkpoint_path[-14:]
        res_saver.restore(sess, model_checkpoint_path)
    else:
        print('No checkpoint file found')
        return None
    return sess

def vis(inputs, outputs, ops, opt, data):
    print('in vis')
    """Training loop"""
    # Unpack inputs, outputs and ops
    x, global_step, stl_params_in, stl_params_in2, stl_params_trg, f_params, f_params2, lr, test_x, test_stl_params_in, val_f_params, is_training = inputs
    loss, merged, test_recon, recons, test_x_in, test_codes, test_codes_trans = outputs
    train_op = ops
    
    # For checkpoints
    gs = 0
    start = time.time()
    
    saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count = {'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=True
    )
    sess = tf.Session(config=config)

    # Initialize variables
    if len(opt['load_path'])>0:
        sess = load_sess(sess, opt['load_path'])
    else:
        print('No model to load')
        return
    
    # check test set
    num_steps = 10#data.test.num_steps(1)
    max_angles_i = 36
    max_angles_j = 36
    fangles_i = np.linspace(0, 2*np.pi, num=max_angles_i)
    fangles_j = np.linspace(0, 2*np.pi, num=max_angles_j)
    
    save_folder = './vis/' + opt['flag'] + '/'
    checkFolder(save_folder)
    for step_i in xrange(num_steps):
        cur_save_folder = save_folder + '%04d/' % step_i
        checkFolder(cur_save_folder)
        print(step_i)
        #pick a random initial transformation
        cur_stl_params_in, _, _, cur_f_params, _ = random_transmats2(1)

        cur_x, cur_y_true = data.test.next_batch(1)
        for j in xrange(max_angles_j):
            cur_save_folder_j_inp = cur_save_folder + '%04d/inp/' % j
            cur_save_folder_j_out = cur_save_folder + '%04d/out/' % j
            cur_save_folder_j_ctrans = cur_save_folder + '%04d/ctrans/' % j
            cur_save_folder_j_cinp = cur_save_folder + '%04d/cinp/' % j
            checkFolder(cur_save_folder_j_inp)
            checkFolder(cur_save_folder_j_out)
            checkFolder(cur_save_folder_j_ctrans)
            checkFolder(cur_save_folder_j_cinp)

            cur_stl_params_j = update_f_params(cur_stl_params_in, fangles_j[j])
            for i in xrange(max_angles_i):
                cur_f_params_i = update_f_params(cur_f_params, fangles_i[i])
                feed_dict = {
                            test_x : cur_x,
                            test_stl_params_in : cur_stl_params_j, 
                            val_f_params: cur_f_params_i,
                            is_training : False
                        }

                cur_y = sess.run(test_recon, feed_dict=feed_dict)
                cur_y = np.reshape(cur_y[0,:].copy(), [28, 28])
                save_name = cur_save_folder_j_out + '/%04d' % (i)
                imsave(save_name + '.png', cur_y) 

                fcodes_trans = open(cur_save_folder_j_ctrans + '%04d.txt' % (i), 'wa')
                cur_codes_trans = sess.run(test_codes_trans, feed_dict=feed_dict)
                np.savetxt(fcodes_trans, cur_codes_trans, fmt='%.6e')
                fcodes_trans.close()

                if i==0:
                    fcodes_inp = open(cur_save_folder_j_cinp + '%04d.txt' % (j), 'wa')
                    cur_codes = sess.run(test_codes, feed_dict=feed_dict)
                    np.savetxt(fcodes_inp, cur_codes, fmt='%.6e')
                    fcodes_inp.close()

                    cur_x_in = sess.run(test_x_in, feed_dict=feed_dict)
                    cur_x_in = np.reshape(cur_x_in[0,:].copy(), [28, 28])
                    save_name = cur_save_folder_j_inp + '/%04d' % (j)
                    imsave(save_name + '.png', cur_x_in) 



def train(inputs, outputs, ops, opt, data):
    """Training loop"""
    # Unpack inputs, outputs and ops
    x, global_step, stl_params_in, stl_params_in2, stl_params_trg, f_params, f_params2, lr, test_x, test_stl_params_in, val_f_params, is_training = inputs
    loss, merged, test_recon, recons, test_x_in, test_codes, test_codes_trans = outputs
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
        sess = load_sess(sess, opt['load_path'])
    else:
        print('Training from scratch')
    
    train_writer = tf.summary.FileWriter(opt['summary_path'], sess.graph)

    # Training loop
    for epoch in xrange(opt['n_epochs']):
        # Learning rate
        exponent = sum([epoch > i for i in opt['lr_schedule']])
        current_lr = opt['lr']*np.power(0.1, exponent)
        
        # Train
        train_loss = 0.
        # Run training steps
        num_steps = data.train.num_examples/opt['mb_size']
        for step_i in xrange(num_steps):
            #_, cur_stl_params_in, cur_stl_params_trg, cur_f_params = random_transmats(opt['mb_size'])
            cur_stl_params_in, cur_stl_params_in2, cur_stl_params_trg, cur_f_params, cur_f_params2 = random_transmats2(opt['mb_size'])

            ops = [global_step, loss, merged, train_op]
            cur_x, cur_y_true = data.train.next_batch(opt['mb_size'])
            
            feed_dict = {
                        x: cur_x,
                        stl_params_in: cur_stl_params_in, 
                        stl_params_in2: cur_stl_params_in2, 
                        stl_params_trg: cur_stl_params_trg, 
                        f_params : cur_f_params, 
                        f_params2 : cur_f_params2, 
                        lr : current_lr,
                        is_training : True
                    }

            gs, l, summary, __ = sess.run(ops, feed_dict=feed_dict)
            train_loss += l
            if step_i % 50==0:
                print('[{:03f}]: train_loss: {:03f}.'.format(float(step_i)/num_steps, l))

            assert not np.isnan(l), 'Model diverged with loss = NaN'

            if step_i % 50==0:
                # Summary writers
                train_writer.add_summary(summary, gs)

        train_loss /= num_steps
        print('[{:03d}]: train_loss: {:03f}.'.format(epoch, train_loss))

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
    elif FLAGS.Mac:
        print('Hello Daniyar!')
        opt['root'] = '/Users/dturmukh'
        dir_ = opt['root'] + '/Documents/Code/harmonicConvolutions/tensorflow1/scale'
    else:
        opt['root'] = '/home/sgarbin'
        dir_ = opt['root'] + '/Projects/harmonicConvolutions/tensorflow1/scale'
    
    opt['mb_size'] = 64
    opt['n_epochs'] = 200
    opt['lr_schedule'] = [20, 40, 60]
    opt['lr'] = 1e-3
    opt['f_params_dim'] = 2
    opt['num_latents'] = opt['f_params_dim']*16
    opt['stl_size'] = 2 # no translation

    #opt['flag'] = 'rotmnist'
    #opt['flag'] = 'rotmnist_vis'
    opt['flag'] = 'rotmnist_conv'
    #opt['flag'] = 'rotmnist_conv_vis'
    opt['summary_path'] = dir_ + '/summaries/autotrain_{:s}'.format(opt['flag'])
    opt['save_path'] = dir_ + '/checkpoints/autotrain_{:s}/'.format(opt['flag'])
    
    ###
    opt['load_path'] = ''
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_rotmnist/'
    #opt['load_path'] = dir_ + '/checkpoints/autotrain_rotmnist_conv/'
    
    #check and clear directories
    checkFolder(opt['summary_path'])
    checkFolder(opt['save_path'])
    checkFolder(dir_ + '/samples/' + opt['flag'])
    
    # Load data
    data = load_data()
    
    # Placeholders
    # batch_size, depth, height, width, in_channels
    x = tf.placeholder(tf.float32, [opt['mb_size'],28*28], name='x')
    stl_params_in  = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],opt['stl_size']], name='stl_params_in')
    stl_params_in2  = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],opt['stl_size']], name='stl_params_in2')
    stl_params_trg = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],opt['stl_size']], name='stl_params_trg')
    f_params = tf.placeholder(tf.float32, [opt['mb_size'], opt['f_params_dim'], opt['f_params_dim']], name='f_params')
    f_params2 = tf.placeholder(tf.float32, [opt['mb_size'], opt['f_params_dim'], opt['f_params_dim']], name='f_params2')

    test_x = tf.placeholder(tf.float32, [1,28*28], name='test_x')
    test_stl_params_in  = tf.placeholder(tf.float32, [1,opt['stl_size'],opt['stl_size']], name='test_stl_params_in')
    val_f_params = tf.placeholder(tf.float32, [1, opt['f_params_dim'], opt['f_params_dim']], name='val_f_params') 
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.placeholder(tf.float32, [], name='lr')
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    
    # generate input and target volumes
    outshape = [28, 28]
    x_img = tf.reshape(x, [-1, 28, 28, 1])

    x_in = tf.reshape(spatial_transform(x_img, stl_params_in, outshape), [-1, 28*28])
    x_in2 = tf.reshape(spatial_transform(x_img, stl_params_in2, outshape), [-1, 28*28])
    x_trg = tf.reshape(spatial_transform(x_img, stl_params_trg, outshape), [-1, 28*28])

    # Build the training model
    if False:
        recons, _, codes1, codes2, recons_logits, _ = autoencoder(x_in, x_in2, opt['num_latents'], f_params, f_params2, is_training)
        
        # Test model
        test_x_img = tf.reshape(test_x, [-1, 28, 28, 1])
        test_x_in = tf.reshape(spatial_transform(test_x_img, test_stl_params_in, outshape), [-1, 28*28])
        test_recon, test_codes, _, _, _, test_codes_trans = autoencoder(test_x_in, test_x_in, opt['num_latents'], val_f_params, val_f_params, is_training, reuse=True)
    else:
        recons, _, codes1, codes2, recons_logits, _ = autoencoder_conv(x_in, x_in2, opt['num_latents'], f_params, f_params2, is_training)
        
        # Test model
        test_x_img = tf.reshape(test_x, [-1, 28, 28, 1])
        test_x_in = tf.reshape(spatial_transform(test_x_img, test_stl_params_in, outshape), [-1, 28*28])
        test_recon, test_codes, _, _, _, test_codes_trans = autoencoder_conv(test_x_in, test_x_in, opt['num_latents'], val_f_params, val_f_params, is_training, reuse=True)


    # LOSS
    loss_codes = 1.*tf.reduce_mean(tf.reduce_sum(tf.square(codes1-codes2), axis=(1)))
    loss = bernoulli_xentropy(x_trg, recons_logits) + loss_codes
    
    # Summaries
    tf_im_summary('recons', tf.reshape(recons, [-1, 28, 28, 1])) 
    tf_im_summary('inputs', tf.reshape(x_in, [-1, 28, 28, 1])) 
    tf_im_summary('targets', tf.reshape(x_trg, [-1, 28, 28, 1])) 
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('LearningRate', lr)
    merged = tf.summary.merge_all()
    
    # Build optimizer
    optim = tf.train.AdamOptimizer(lr)
    #optim = tf.train.MomentumOptimizer(lr, momentum=0.1, use_nesterov=True)
    train_op = optim.minimize(loss, global_step=global_step)
    
    # Set inputs, outputs, and training ops
    inputs = [x, global_step, stl_params_in, stl_params_in2, stl_params_trg, f_params, f_params2, lr, test_x, test_stl_params_in, val_f_params, is_training]
    ops = [train_op]
    outputs = [loss, merged, test_recon, recons, test_x_in, test_codes, test_codes_trans]

    if FLAGS.VIS:
        return vis(inputs, outputs, ops, opt, data)
    # Train
    return train(inputs, outputs, ops, opt, data)


if __name__ == '__main__':
    tf.app.run()
