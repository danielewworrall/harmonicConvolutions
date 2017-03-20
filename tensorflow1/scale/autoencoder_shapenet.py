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
flags.DEFINE_integer('save_step', 10, 'Interval (epoch) for which to save')
flags.DEFINE_boolean('Daniel', False, 'Daniel execution environment')
flags.DEFINE_boolean('Sleepy', False, 'Sleepy execution environment')
flags.DEFINE_boolean('Dopey', True, 'Dopey execution environment')
flags.DEFINE_boolean('DaniyarSleepy', False, 'Dopey execution environment')
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
    
    l0 = convlayer(0, x,  3, 1,      8,  1, reuse, is_training=is_training)                   #  48-46
    l1 = convlayer(1, l0, 3, 8,     16,  2, reuse, is_training=is_training)                   #  46-24
    l2 = convlayer(2, l1, 3, 16,    16,  1, reuse, is_training=is_training)                  #  24-24
    l3 = convlayer(3, l2, 3, 16,    32,  2, reuse, is_training=is_training)                   #  24-12
    l4 = convlayer(4, l3, 3, 32,    32,  1, reuse, is_training=is_training)                  #  12-12
    l5 = convlayer(5, l4, 3, 32,    64,  2, reuse, is_training=is_training)                   #  12-6
    l6 = convlayer(6, l5, 3, 64,    64,  1, reuse, is_training=is_training)                   #  6-6
    l7 = convlayer(7, l6, 6, 64,   512,  1, reuse, is_training=is_training, padding='VALID')  #  6-1 
    codes = convlayer(8, l7, 1, 512, num_latents, 1, reuse, nonlin=tf.identity, is_training=is_training, dobn=False) # 1 -> 1
    return codes


def decoder(codes, is_training, reuse=False):
    num_latents = codes.get_shape()[-1]


    l1 = upconvlayer(1,             codes, 1, num_latents, 512,   1, reuse, is_training=is_training) 
    l2 = upconvlayer_tr(2,          l1,    6, 512,         64,    6, 6, reuse, is_training=is_training)
    l3 = upconvlayer(3,             l2,    3, 64,          64,    6, reuse, is_training=is_training)
    l4 = upconvlayer(4,             l3,    3, 64,          64,   12, reuse, is_training=is_training)
    l5 = upconvlayer(5,             l4,    3, 64,          32,   12, reuse, is_training=is_training)
    l6 = upconvlayer(6,             l5,    3, 32,          32,   24, reuse, is_training=is_training)
    l7 = upconvlayer(7,             l6,    3, 32,          16,   24, reuse, is_training=is_training)
    l8 = upconvlayer(8,             l7,    3, 16,          16,   48, reuse, is_training=is_training)
    recons_logits = upconvlayer(9,  l8,    3, 16,          1,    48, reuse, is_training=is_training, nonlin=tf.identity, dobn=False)
    recons = tf.sigmoid(recons_logits)
    return recons, recons_logits

def ssim3d(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    def avg_pool3d(x, kw, stride, padding='VALID'):
        return tf.nn.avg_pool3d(x,
                          ksize=[1, kw, kw, kw, 1],
                          strides=[1, stride, stride, stride, 1],
                          padding=padding)

    x = avg_pool3d(x, 3, 1, 'VALID')
    y = avg_pool3d(y, 3, 1, 'VALID')

    mu_x = avg_pool3d(x, 3, 1, 'VALID')
    mu_y = avg_pool3d(y, 3, 1, 'VALID')

    sigma_x  = avg_pool3d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = avg_pool3d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = avg_pool3d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    cost = tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    print(cost)
    cost = tf.reduce_sum(cost, axis=(1,2,3,4))
    return cost

def diffl1(x,y):
    batch_size = x.get_shape().as_list()[0]
    x = tf.reshape(x, [batch_size, -1])
    y = tf.reshape(y, [batch_size, -1])
    return (tf.reduce_sum(tf.abs(x-y), axis=1))

def bernoulli_xentropy(target, output):
    """Cross-entropy for Bernoulli variables"""
    target = 3*target-1
    output = 0.8999*output + 0.1000
    wx_entropy = -(98.0 * target * tf.log(output) + 2.0*(1.0 - target) * tf.log(1.0 - output))/100.0
    return (tf.reduce_sum(wx_entropy, axis=(1,2,3,4)))


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

    # scale and rot because inverse warp in stl
    stl_transmat_inp = np.matmul(inp_3dscalemat, inp_3drotmat)
    stl_transmat_trg = np.matmul(trg_3dscalemat, trg_3drotmat)
    
    f_params_inp = np.zeros([batch_size, 3, 3])
    # was like this:
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
        current_lr = opt['lr']*np.power(0.1, exponent)
        
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

            print('[{:03f}]: {:03f}'.format(float(step_i)/num_steps, l))

            assert not np.isnan(l), 'Model diverged with loss = NaN'

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
    
    opt['mb_size'] = 16
    opt['n_epochs'] = 200
    opt['lr_schedule'] = [190]
    opt['lr'] = 1e-3

    opt['vol_size'] = [32,32,32]
    pad_size = 8#int(np.ceil(np.sqrt(3)*opt['vol_size'][0]/2)-opt['vol_size'][0]/2)
    opt['outsize'] = [i + 2*pad_size for i in opt['vol_size']]
    stl = AffineVolumeTransformer(opt['outsize'])
    opt['color_chn'] = 1
    opt['stl_size'] = 3 # no translation
    # TODO
    opt['f_params_dim'] = 3# + 2*3 # rotation matrix is 3x3 and we have 3 axis scalings implemented as 2x2 rotations
    opt['num_latents'] = opt['f_params_dim']*100


    opt['flag'] = 'shapenet_l1ssim'
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
    recons, codes, recons_logits = autoencoder(x_in, opt['num_latents'], f_params, is_training)
    
    # Test model
    test_x_in = spatial_transform(stl, test_x, test_stl_params_in, paddings)
    test_recon, __, _ = autoencoder(test_x_in, opt['num_latents'], val_f_params, is_training, reuse=True)
    
    # LOSS
    #loss = tf.reduce_mean(bernoulli_xentropy(x_trg, recons))
    #loss = tf.reduce_mean(0.5*diffl1(x_trg, recons) + 0.5*ssim3d(x_trg, recons))
    loss = tf.reduce_mean(0.5*bernoulli_xentropy(x_trg, recons) + 0.5*ssim3d(x_trg, recons))
    
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
