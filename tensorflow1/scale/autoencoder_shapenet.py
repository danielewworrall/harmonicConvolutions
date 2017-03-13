import os
import sys
import time
import glob
sys.path.append('../')
import numpy as np
import tensorflow as tf

### local files ######
import equivariant_loss as el
import shapenet_loader
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
flags.DEFINE_integer('save_step', 50, 'Interval (epoch) for which to save')
flags.DEFINE_boolean('Daniel', False, 'Daniel execution environment')
flags.DEFINE_boolean('Sleepy', False, 'Sleepy execution environment')
flags.DEFINE_boolean('Dopey', True, 'Dopey execution environment')
##---------------------

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
            code = encoder(x, num_latents, is_training, reuse=reuse)
        with tf.variable_scope("feature_transformer", reuse=reuse) as scope:
            code_shape = code.get_shape().as_list()
            batch_size = code_shape[0]
            code = tf.reshape(code, [batch_size, -1])
            matrix_shape = [batch_size, code.get_shape()[1]]
            # TODO
            code_transformed = el.feature_transform_matrix_n(code, matrix_shape, f_params)
            code_transformed = tf.reshape(code_transformed, code_shape)
        with tf.variable_scope("decoder", reuse=reuse) as scope:
            recon = decoder(code_transformed, is_training, reuse=reuse)
    return recon, code

def variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False), trainable=True):
    #tf.random_normal_initializer(stddev=0.0001)
    #tf.constant_initializer(0.0)
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var


def encoder(x, num_latents, is_training, reuse=False):
    """Encoder with conv3d"""
    
    def convlayer(i, inp, ksize, inpdim, outdim, stride, reuse):
        scopename = 'conv_layer' + str(i)
        print(scopename)
        print(' input:', inp)
        with tf.variable_scope(scopename) as scope:
            if reuse:
                scope.reuse_variables()
            kernel = variable(scopename + '_kernel', [ksize, ksize, ksize, inpdim, outdim])
            bias = variable(scopename + '_bias', [outdim], tf.constant_initializer(0.0))
            linout = bias + tf.nn.conv3d(inp, kernel, strides=[1, stride, stride, stride, 1], padding='SAME')
            bnout = bn5d(linout, is_training, reuse=reuse)
            out = tf.nn.elu(bnout, name=scopename + 'elu')
        print(' out:', out)
        return out

    l1 = convlayer(1, x, 3, 1, 16, 2, reuse) # 56
    l2 = convlayer(2, l1, 3, 16, 32, 2, reuse) # 28
    l3 = convlayer(3, l2, 3, 32, 32, 2, reuse) # 14
    l4 = convlayer(4, l3, 3, 32, 64, 2, reuse) # 7
    l5 = convlayer(5, l4, 3, 64, 64, 2, reuse) # 4
    l6 = convlayer(6, l5, 3, 64, 128, 2, reuse) # 2
    codes = convlayer(7, l6, 2, 128, num_latents, 2, reuse) # 1
    return codes


def decoder(code, is_training, reuse=False):

    #def convlayer(i, inp, ksize, inpdim, outdim, stride, reuse):
    #    scopename = 'conv_layer' + str(i)
    #    with tf.variable_scope(scopename) as scope:
    #        if reuse:
    #            scope.reuse_variables()
    #        kernel = variable(scopenale + '_kernel', [ksize, ksize, ksize, inpdim, outdim])
    #        bias = variable(scopenale + '_bias', [outdim], tf.constant_initializer(0.0))
    #        linout = bias + tf.nn.conv3d(inp, kernel, strides=stride, padding='SAME')
    #        bnout = ops.get_batchnorm(self.batchnorms.get(scopename), lin, name=scopename)
    #        out = tf.nn.elu(bnout, name=scopename + 'elu')
    #    return out

    #l1 = convlayer(1, x, 3, 1, 16, 2, reuse) # 56
    #l2 = convlayer(2, l1, 3, 16, 32, 2, reuse) # 28
    #l3 = convlayer(3, l2, 3, 32, 32, 2, reuse) # 14
    #l4 = convlayer(4, l3, 3, 32, 64, 2, reuse) # 7
    #l5 = convlayer(5, l4, 3, 64, 64, 2, reuse) # 4
    #l6 = convlayer(6, l5, 3, 64, 128, 2, reuse) # 2
    #recons = convlayer(7, l6, 2, 128, flags.num_latents, 2, reuse) # 1
    #return recons
    return code


def bernoulli_xentropy(x, test_recon):
    """Cross-entropy for Bernoulli variables"""
    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=test_recon)
    return tf.reduce_mean(tf.reduce_sum(x_entropy, axis=(1,2)))


def spatial_transform(stl, x, params, paddings):
    x_padded = tf.pad(x, paddings, mode='CONSTANT')
    x_in = stl.transform(x_padded, params)
    return x_in

def update_f_params(f_params, val):
    # TODO
    #r0 = np.random.rand() > 0.5
    #r1 = np.random.rand() > 0.5
    #r2 = np.random.rand() > 0.5
    #fv_ = np.vstack([2.*r0*fv, r1*fv, r2*fv]).T
    return f_params

def random_transmats(mb_size, fv):
    """ Random rotation in 3D
    """
    # TODO
    tp_in, _ = el.random_transform(opt['mb_size'], opt['vol_size'])

    t_params = []
    f_params = []
    
    # TODO
    if fv is None:
        fv = np.pi*np.random.rand(mb_size, 3)
        fv[:,0] = 2.*fv[:,0]
    
    for i in xrange(mb_size):
        # Anisotropically scaled and rotated
        rot = np.array([[np.cos(fv[i,0]), -np.sin(fv[i,0])],
                        [np.sin(fv[i,0]), np.cos(fv[i,0])]])
        scale = np.array([[scale_(fv[i,1],0.8,1.2),0.],[0., scale_(fv[i,2],0.8,1.2)]])
        transform = np.dot(scale, rot)
        # Compute transformer matrices
        t_params.append(el.get_t_transform_n(transform, (imsh[0],imsh[1])))
        f_params.append(el.get_f_transform_n(fv[i,:]))

    return np.vstack(t_params), np.stack(f_params, axis=0)


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
    x, global_step, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, f_params_val, is_training = inputs
    loss, merged, test_recon = outputs
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
            
            feed_dict = {
                        x: data.train.next_batch(opt['mb_size']),
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
        train_loss /= num_steps
        print('[{:03d}]: {:03f}'.format(epoch, train_loss))
        
        # Save model
        if epoch % FLAGS.save_step == 0:
            path = saver.save(sess, opt['save_path'], epoch)
            print('Saved model to ' + path)
        
        # Validation
        if epoch % 1 == 0:
            recons = []
            max_angles = 20
            #pick a random initial transformation
            cur_stl_params_in, _, cur_f_params = random_transmats(opt['mb_size'])
            sample = data.validation.next_batch(opt['mb_size'])
            fparams = np.linspace(0., np.pi, num=max_angles)

            for j in xrange(max_angles):
                cur_f_params_j = update_f_params(cur_f_params, 0, fparams[j])
                for i in xrange(max_angles):
                    cur_f_params_ji = update_f_params(cur_f_params_j, 1, fparams[i])

                    feed_dict = {
                                test_x : sample,
                                test_stl_params_in : cur_stl_params_in, 
                                f_params_val : cur_f_params_ji,
                                is_training : False
                            }

                    y = sess.run(test_recon, feed_dict=feed_dict)
                    recons.append(y[0,:,:,:,:].copy())
            
            samples_ = np.stack(recons)
            print('validation samples shape', samples_.shape)

            tile_image = np.reshape(samples_, [max_angles*max_angles, opt['vol_size'][0], opt['vol_size'][1], opt['vol_size[2]'], opt['color_chn']])
            tile_image = np.sum(tile_image, axis=5)
            tile_image_d = np.sum(tile_image, axis=1)
            tile_image_h = np.sum(tile_image, axis=2)
            tile_image_w = np.sum(tile_image, axis=3)

            tile_image_d /= np.sum(tile_image_d)
            tile_image_h /= np.sum(tile_image_h)
            tile_image_w /= np.sum(tile_image_w)

            def tile_batch(batch, tile_w=1, tile_h=1):
                assert tile_w * tile_h == batch.shape[0], 'tile dimensions inconsistent'
                batch = np.split(batch, tile_w*tile_h, axis=0)
                batch = [np.concatenate(batch[i*tile_w:(i+1)*tile_w], 2) for i in range(tile_h)]
                batch = np.concatenate(batch, 1)
                batch = batch[0,:,:]
                return batch

            tile_image_d = 1.0 - tile_batch(tile_image_d, max_angles, max_angles)
            tile_image_h = 1.0 - tile_batch(tile_image_h, max_angles, max_angles)
            tile_image_w = 1.0 - tile_batch(tile_image_w, max_angles, max_angles)

            save_name = './samples/vae/image_%04d' % epoch
            skio.imsave(save_name + '_d.png', tile_image_d) 
            skio.imsave(save_name + '_h.png', tile_image_h) 
            skio.imsave(save_name + '_w.png', tile_image_w) 

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
    else:
        opt['root'] = '/home/sgarbin'
        dir_ = opt['root'] + '/Projects/harmonicConvolutions/tensorflow1/scale'
    
    opt['mb_size'] = 128
    opt['n_channels'] = 10
    opt['n_epochs'] = 2000
    opt['lr_schedule'] = [50, 75]
    opt['lr'] = 1e-3

    opt['vol_size'] = [32,32,32]
    pad_size = int(np.ceil(np.sqrt(3)*opt['vol_size'][0]/2)-opt['vol_size'][0]/2)
    opt['outsize'] = [i + pad_size for i in opt['vol_size']]
    stl = AffineVolumeTransformer(opt['outsize'])
    opt['color_chn'] = 1
    opt['num_latents'] = 64
    opt['stl_param_dim'] = stl.param_dim


    flag = 'vae'
    opt['summary_path'] = dir_ + '/summaries/autotrain_{:s}'.format(flag)
    opt['save_path'] = dir_ + '/checkpoints/autotrain_{:s}/model.ckpt'.format(flag)
    
    #check and clear directories
    checkFolder(opt['summary_path'])
    checkFolder(opt['save_path'])
    removeAllFilesInDirectory(opt['summary_path'], '.*')
    removeAllFilesInDirectory(opt['save_path'], '.*')
    
    # Load data
    data = load_data()
    
    # Placeholders
    # batch_size, depth, height, width, in_channels
    x = tf.placeholder(tf.float32, [opt['mb_size'],opt['vol_size'][0],opt['vol_size'][1],opt['vol_size'][2], opt['color_chn']], name='x')
    stl_params_in  = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_param_dim']], name='stl_params_in')
    stl_params_trg = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_param_dim']], name='stl_params_trg')
    f_params     = tf.placeholder(tf.float32, [opt['mb_size'],opt['stl_param_dim'],opt['stl_param_dim']], name='f_params')

    test_x = tf.placeholder(tf.float32, [1,opt['vol_size'][0],opt['vol_size'][1],opt['vol_size'][2],opt['color_chn']], name='test_x')
    test_stl_params_in  = tf.placeholder(tf.float32, [1,opt['stl_param_dim']], name='test_stl_params_in')
    val_f_params = tf.placeholder(tf.float32, [1,opt['stl_param_dim'],opt['stl_param_dim']], name='val_f_params') 
    paddings = tf.convert_to_tensor(np.array([[0,0], [pad_size,pad_size], [pad_size,pad_size], [pad_size, pad_size], [0,0]]))
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.placeholder(tf.float32, [], name='lr')
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    
    # Build the training model
    x_in = spatial_transform(stl, x, stl_params_in, paddings)
    x_trg = spatial_transform(stl, x, stl_params_trg, paddings)
    recon, codes = autoencoder(x_in, opt['num_latents'], f_params, is_training)
    
    # Test model
    test_x_in = spatial_transform(stl, test_x, test_stl_params_in, paddings)
    test_recon, __ = autoencoder(test_x_in, opt['num_latents'], val_f_params, is_training, reuse=True)
    
    # LOSS
    loss = bernoulli_xentropy(x_trg, recon)
    
    # Summaries
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Loss_Img', nll)
    tf.summary.scalar('LearningRate', lr)
    merged = tf.summary.merge_all()
    
    # Build optimizer
    optim = tf.train.AdamOptimizer(lr)
    train_op = optim.minimize(loss, global_step=global_step)
    
    # Set inputs, outputs, and training ops
    inputs = [x, global_step, stl_params_in, stl_params_trg, f_params, lr, test_x, test_stl_params_in, f_params_val, is_training]
    outputs = [loss, merged, test_recon]
    ops = [train_op]
    
    # Train
    return train(inputs, outputs, ops, opt, data)

if __name__ == '__main__':
    tf.app.run()
