'''Run BSD500'''

import argparse
import os
import shutil
import sys
import time
sys.path.append('../')

import cPickle as pkl
import numpy as np
import skimage.io as skio
import tensorflow as tf

from io_helpers import download_dataset, load_pkl, pklbatcher
from BSD_model import hnet_bsd, vgg_bsd


def make_dirs(args, directory):
   if directory is not None:
      if not os.path.exists(directory):
         os.makedirs(directory)
         print('Created {:s}'.format(directory))
      else:
         if not args.delete_existing:
            raw_input('{:s} already exists: Press ENTER to overwrite contents.'.format(directory))
         shutil.rmtree(directory)
         os.makedirs(directory)


def settings(args):
   """Load the data and default settings"""
   data = load_pkl('./data', 'bsd_pkl_float', prepend='')
   # Default configuration
   if args.default_settings:
      args.n_epochs = 250
      args.batch_size = 10
      args.learning_rate = 1e-2
      args.std_mult = 1.
      args.delay = 8
      args.phase_preconditioner = 3.4
      args.filter_gain = 2
      args.filter_size = 3
      args.n_rings = 2
      args.n_filters = 7
      args.display_step = len(data['train_x'])/46
      args.dim = 321
      args.dim2 = 481

      args.n_channels = 3
      args.lr_div = 10.
      args.augment = True
      args.sparsity = True

      args.test_path = args.save_name
      args.log_path = os.path.join('./logs/', args.test_path)
      args.checkpoint_path = os.path.join('./checkpoints/', args.test_path)
      args.test_path = os.path.join('./', args.test_path)

      make_dirs(args, args.test_path)
      make_dirs(args, args.log_path)
      make_dirs(args, args.checkpoint_path)

   return args, data


def minibatcher(inputs, targets, batchsize, shuffle=False):
   assert len(inputs) == len(targets)
   if shuffle:
      indices = np.arange(len(inputs))
      np.random.shuffle(indices)
   for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
      if shuffle:
         excerpt = indices[start_idx:start_idx + batchsize]
      else:
         excerpt = slice(start_idx, start_idx + batchsize)
      yield inputs[excerpt], targets[excerpt]


def get_learning_rate(opt, current, best, counter, learning_rate):
   """If have not seen accuracy improvement in delay epochs, then divide
   learning rate by 10
   """
   if current > best:
      best = current
      counter = 0
   elif counter > opt['delay']:
      learning_rate = learning_rate / 10.
      counter = 0
   else:
      counter += 1
   return (best, counter, learning_rate)


def sparsity_regularizer(x, sparsity):
   """Define a sparsity regularizer"""
   q = tf.reduce_mean(tf.nn.sigmoid(x))
   return -sparsity*tf.log(q) - (1-sparsity)*tf.log(1-q)


def main(args):
   """The magic happens here"""
   print('Setting up')
   tf.reset_default_graph()
   # SETUP AND LOAD DATA
   print('...Loading settings and data')
   args, data = settings(args)

   # BUILD MODEL
   ## Placeholders
   print('...Creating network input')
   x = tf.placeholder(tf.float32, [args.batch_size,None,None,3], name='x')
   y = tf.placeholder(tf.float32, [args.batch_size,None,None,1], name='y')
   learning_rate = tf.placeholder(tf.float32, name='learning_rate')
   train_phase = tf.placeholder(tf.bool, name='train_phase')

   ## Construct model
   print('...Constructing model')
   if args.mode == 'baseline':
      pred = vgg_bsd(args, x, train_phase)
   elif args.mode == 'hnet':
      pred = hnet_bsd(args, x, train_phase)
   else:
      print('Must execute script with valid --mode flag: "hnet" or "baseline"')
      sys.exit(-1)

   # Print number of parameters
   n_vars = 0
   for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      n_vars += np.prod(var.get_shape().as_list())
   print('...Number of parameters: {:d}'.format(n_vars))

   print('...Building loss')
   loss = 0.
   beta = 1-tf.reduce_mean(y)
   pw = beta / (1. - beta)
   for key in pred.keys():
      pred_ = pred[key]
      loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, pred_, pw))
      # Sparsity regularizer
      loss += args.sparsity*sparsity_regularizer(pred_, 1-beta)
   ## Optimizer
   print('...Building optimizer')
   optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
   grads_and_vars = optim.compute_gradients(loss)
   modified_gvs = []
   for g, v in grads_and_vars:
      print(v.name)
      print v, g
      if 'phase' in v.name:
         print g
         g = args.phase_preconditioner*g
      modified_gvs.append((g, v))
   train_op = optim.apply_gradients(modified_gvs)

   # TRAIN
   print('...Initializing variables')
   init = tf.global_variables_initializer()
   init_local = tf.local_variables_initializer()

   # Configure tensorflow session
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   config.log_device_placement = False

   print('TRAINING')
   lr = args.learning_rate
   saver = tf.train.Saver()
   with tf.Session(config=config) as sess:
      sess.run([init, init_local], feed_dict={train_phase : True})

      start = time.time()
      epoch = 0
      while epoch < args.n_epochs:
         anneal = 0.1 + np.minimum(epoch/30.,1.)
         # Training steps
         batcher = pklbatcher(data['train_x'], data['train_y'], args.batch_size, shuffle=True, augment=True)
         train_loss = 0.
         for i, (X, Y, __) in enumerate(batcher):
            feed_dict = {x: X, y: Y, learning_rate: lr, train_phase: True}
            __, l = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += l
            sys.stdout.write('{:d}/{:d}\r'.format(i, len(data['train_x'].keys())/args.batch_size))
            sys.stdout.flush()
         train_loss /= (i+1.)

         print('[{:04d} | {:0.1f}] Loss: {:04f}, Learning rate: {:.2e}'.format(epoch,
            time.time() - start, train_loss, lr))


         if epoch % args.save_step == 0:
            # Validate
            save_path = args.test_path + '/T_' + str(epoch)
            if not os.path.exists(save_path):
               os.mkdir(save_path)
            generator = pklbatcher(data['valid_x'], data['valid_y'],
                                       args.batch_size, shuffle=False,
                                       augment=False, img_shape=(args.dim, args.dim2))
            # Use sigmoid to map to [0,1]
            bsd_map = tf.nn.sigmoid(pred['fuse'])
            j = 0
            for batch in generator:
               batch_x, batch_y, excerpt = batch
               output = sess.run(bsd_map, feed_dict={x: batch_x, train_phase: False})
               for i in xrange(output.shape[0]):
                  save_name = save_path + '/' + str(excerpt[i]).replace('.jpg','.png')
                  im = output[i,:,:,0]
                  im = (255*im).astype('uint8')
                  if data['valid_x'][excerpt[i]]['transposed']:
                     im = im.T
                  skio.imsave(save_name, im)
                  j += 1
            print('Saved predictions to: %s' % (save_path,))

         # Updates to the training scheme
         if epoch % 40 == 39:
            lr = lr / 10.
         epoch += 1

         # Save model
         saver.save(sess, args.checkpoint_path + 'model.ckpt')
   return train_loss, n_vars


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--mode", help="model to run {hnet,baseline}", default="hnet")
   parser.add_argument("--save_name", help="name of the checkpoint path", default="my_model")
   parser.add_argument("--data_dir", help="data directory", default='./data')
   parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
   parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool, default=False)
   parser.add_argument("--delete_existing", help="delete the existing auxilliary files", type=bool, default=True)
   main(parser.parse_args())
