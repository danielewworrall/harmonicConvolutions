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
flags.DEFINE_boolean('DaniyarSleepy', False, 'Dopey execution environment')
flags.DEFINE_boolean('Mac', True, 'Mac execution environment')
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

def variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False), trainable=True):
    #tf.constant_initializer(0.0)
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

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

def encoder(x, num_latents, is_training, reuse=False):
    """Encoder MLP"""
    l1 = mlplayer(0, x, 784, 512, is_training=is_training, reuse=reuse, dobn=True)
    l2 = mlplayer(1, l1, 512, 512, is_training=is_training, reuse=reuse, dobn=True)
    codes = mlplayer(2, l2, 512, num_latents, is_training=is_training, reuse=reuse, dobn=False)
    return codes


def decoder(z, is_training, reuse=False):
    """Encoder MLP"""
    l2 = mlplayer(3, z, z.get_shape()[1], 512, is_training=is_training, reuse=reuse, dobn=True)
    l1 = mlplayer(4, l2, 512, 512, is_training=is_training, reuse=reuse, dobn=True)
    recons_logits = mlplayer(5, l1, 512, 784, is_training=is_training, reuse=reuse, dobn=False, nonlin=tf.identity)
    recons = tf.sigmoid(recons_logits)
    return recons, recons_logits

def autoencoder(x, num_latents, f_params, is_training, reuse=False):
    print(x)
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


def bernoulli_xentropy(x, recon_test):
    """Cross-entropy for Bernoulli variables"""
    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_test)
    return tf.reduce_mean(tf.reduce_sum(x_entropy, axis=(1)))


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
        model_checkpoint_path = load_path + ckpt.model_checkpoint_path[-13:]
        res_saver.restore(sess, model_checkpoint_path)
    else:
        print('No checkpoint file found')
        return None
    return sess

def vis(inputs, outputs, ops, opt, data):
    print('in vis')
    """Training loop"""
    # Unpack inputs, outputs and ops
    x, global_step, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training = inputs
    loss, merged, test_recon, recons, test_x_in = outputs
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
    num_steps = 2#data.test.num_steps(1)
    for step_i in xrange(num_steps):
        print(step_i)
        val_recons = []
        val_vox = []
        #pick a random initial transformation
        _, cur_stl_params_in, _, cur_f_params = random_transmats(1)

        max_angles = 36
        cur_x, cur_y_true = data.test.next_batch(1)
        fangles = np.linspace(0, 2*np.pi, num=max_angles)
        
        save_folder = './vis/' + opt['flag'] + '/'
        checkFolder(save_folder)
        j = 0
        for i in xrange(max_angles):
            cur_stl_params_j = update_f_params(cur_stl_params_in, fangles[j])
            cur_f_params_i = update_f_params(cur_f_params, fangles[i])
            print(i, cur_f_params_i)
            feed_dict = {
                        test_x : cur_x,
                        test_stl_params_in : cur_stl_params_j, 
                        val_f_params: cur_f_params_i,
                        is_training : False
                    }

            cur_y = sess.run(test_recon, feed_dict=feed_dict)
            val_recons.append(np.reshape(cur_y[0,:].copy(), [28, 28]))
            save_name = save_folder + '/image_%04d_%03d' % (step_i, i)
            imsave(save_name + '.png', val_recons[i]) 

            if i==0:
                cur_x_in = sess.run(test_x_in, feed_dict=feed_dict)
                cur_x_in = np.reshape(cur_x_in[0,:].copy(), [28, 28])
                save_name = save_folder + '/input_%04d' % step_i
                imsave(save_name + '.png', cur_x_in) 



def train(inputs, outputs, ops, opt, data):
    """Training loop"""
    # Unpack inputs, outputs and ops
    x, global_step, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training = inputs
    loss, merged, test_recon, recons, test_x_in = outputs
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
            cur_stl_params_aug, cur_stl_params_in, cur_stl_params_trg, cur_f_params = random_transmats(opt['mb_size'])
            ops = [global_step, loss, merged, train_op]
            cur_x, cur_y_true = data.train.next_batch(opt['mb_size'])
            
            feed_dict = {
                        x: cur_x,
                        stl_params_in: cur_stl_params_in, 
                        stl_params_trg: cur_stl_params_trg, 
                        f_params : cur_f_params, 
                        lr : current_lr,
                        is_training : True
                    }

            gs, l, summary, __ = sess.run(ops, feed_dict=feed_dict)
            train_loss += l
            if step_i % 10==0:
                print('[{:03f}]: train_loss: {:03f}.'.format(float(step_i)/num_steps, l))

            assert not np.isnan(l), 'Model diverged with loss = NaN'

            if step_i % 10==0:
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
    
    opt['mb_size'] = 32
    opt['n_epochs'] = 100
    opt['lr_schedule'] = [25, 50, 75]
    opt['lr'] = 1e-3
    opt['f_params_dim'] = 2
    opt['num_latents'] = opt['f_params_dim']*16
    opt['stl_size'] = 2 # no translation

    opt['flag'] = 'rotmnist_vis'
    opt['summary_path'] = dir_ + '/summaries/autotrain_{:s}'.format(opt['flag'])
    opt['save_path'] = dir_ + '/checkpoints/autotrain_{:s}/'.format(opt['flag'])
    
    ###
    #opt['load_path'] = ''
    opt['load_path'] = dir_ + '/checkpoints/autotrain_rotmnist/'
    
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
    stl_params_trg = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_size'],opt['stl_size']], name='stl_params_trg')
    f_params = tf.placeholder(tf.float32, [opt['mb_size'], opt['f_params_dim'], opt['f_params_dim']], name='f_params')

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
    x_trg = tf.reshape(spatial_transform(x_img, stl_params_trg, outshape), [-1, 28*28])

    # Build the training model
    recons, codes, recons_logits = autoencoder(x_in, opt['num_latents'], f_params, is_training)
    
    # Test model
    test_x_img = tf.reshape(test_x, [-1, 28, 28, 1])
    test_x_in = tf.reshape(spatial_transform(test_x_img, test_stl_params_in, outshape), [-1, 28*28])
    test_recon, test_codes, _ = autoencoder(test_x_in, opt['num_latents'], val_f_params, is_training, reuse=True)

    # LOSS
    loss = bernoulli_xentropy(x_trg, recons_logits)
    
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
    inputs = [x, global_step, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, val_f_params, is_training]
    ops = [train_op]
    outputs = [loss, merged, test_recon, recons, test_x_in]

    if FLAGS.VIS:
        return vis(inputs, outputs, ops, opt, data)
    # Train
    return train(inputs, outputs, ops, opt, data)


if __name__ == '__main__':
    tf.app.run()
