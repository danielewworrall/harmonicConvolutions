'''Boundary detection---suff diff to everything else to require own file'''

import os
import sys
import time

from equivariant import deep_bsd

import os
import sys
import time

import cv2
import equivariant
import numpy as np
import scipy as sp
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import tensorflow as tf

import input_data

from equivariant import *
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import misc
from steer_conv import *

###HELPER FUNCTIONS------------------------------------------------------------------
def get_loss(opt, pred, y):
    """Pred is a dist of feature maps and so is y"""
    cost = 0.
    for key in pred.keys():
        y_ = y #[key]
        pred_ = pred[key]
        pw = 1-tf.reduce_mean(y_)
        # side-weight/fusion loss
        cost += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred_, y_, pw))
    print('  Constructed loss')
    return cost

def get_io_placeholders(opt):
	"""Return placeholders for classification/regression"""
	io_x = tf.placeholder(tf.float32, [opt['batch_size'],opt['dim'],opt['dim2'],3])
	io_y = tf.placeholder(tf.float32, [opt['batch_size'],opt['dim'],opt['dim2'],1], name='y')
	return io_x, io_y

def build_optimizer(cost, lr, opt):
	"""Apply the psi_precponditioner"""
	mmtm = tf.train.MomentumOptimizer
	optim = mmtm(learning_rate=lr, momentum=opt['momentum'], use_nesterov=True)
	
	grads_and_vars = optim.compute_gradients(cost)
	modified_gvs = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			g = opt['psi_preconditioner']*g
		modified_gvs.append((g, v))
	optimizer = optim.apply_gradients(modified_gvs)
	print('  Optimizer built')
	return optimizer

def get_evaluation(pred, y, opt):
	correct_pred = tf.equal(pred, y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy

def build_feed_dict(opt, io, batch, lr, pt, lr_, pt_):
	'''Build a feed_dict appropriate to training regime'''
	batch_x, batch_y = batch
	fd = {lr : lr_, pt : pt_}
	bs = opt['batch_size']
	for g in xrange(len(opt['deviceIdxs'])):
		fd[io['x'][g]] = batch_x[g*bs:(g+1)*bs,:]
		fd[io['y'][g]] = batch_y[g*bs:(g+1)*bs]
	return fd

##### TRAINING LOOPS #####
def loop(mode, sess, io, opt, data, cost, acc, lr, lr_, pt, optim=None, step=0):
	"""Run a loop"""
	X = data[mode+'_x']
	Y = data[mode+'_y']
	is_training = (mode=='train')
	n_GPUs = len(opt['deviceIdxs'])
	generator = minibatcher(X, Y, n_GPUs*opt['batch_size'],
							shuffle=is_training, augment=opt['augment'],
							img_shape=(opt['dim'], opt['dim']),
							crop_shape=opt['crop_shape'])
	cost_total = 0.
	acc_total = 0.
	for i, batch in enumerate(generator):
		fd = build_feed_dict(opt, io, batch, lr, pt, lr_, is_training)
		if mode == 'train':
			__, cost_, acc_ = sess.run([optim, cost, acc], feed_dict=fd)
		else:
			cost_, acc_ = sess.run([cost, acc], feed_dict=fd)
		if step % opt['display_step'] == 0:
			print('  ' + mode + ' Acc.: %f' % acc_)
		cost_total += cost_
		acc_total += acc_
		step += 1
	return cost_total/(i+1.), acc_total/(i+1.), step

def construct_model_and_optimizer(opt, io, lr, pt):
	"""Build the model and an single/multi-GPU optimizer"""
	if len(opt['deviceIdxs']) == 1:
		pred = opt['model'](opt, io['x'][0], pt)
		loss = get_loss(opt, pred, io['y'][0])
		accuracy = get_evaluation(pred, io['y'][0], opt)
		train_op = build_optimizer(loss, lr, opt)
	return loss, accuracy, train_op

def train_model(opt, data):
    """Generalized training function
    
    opt: dict of options
    data: dict of numpy data
    """
    n_GPUs = len(opt['deviceIdxs'])
    print('Using Multi-GPU Model with %d devices.' % n_GPUs)
    # Make placeholders
    io = {}
    io['x'] = []
    io['y'] = []
    for g in opt['deviceIdxs']:
        with tf.device('/gpu:%d' % g):
            io_x, io_y = get_io_placeholders(opt)
            io['x'].append(io_x)
            io['y'].append(io_y)
    lr = tf.placeholder(tf.float32, name='learning_rate')
    pt = tf.placeholder(tf.bool, name='phase_train')
    
    # Construct model and optimizer
    loss, accuracy, train_op = construct_model_and_optimizer(opt, io, lr, pt)
    
    # Initializing the variables
    init = tf.initialize_all_variables()
    if opt['combine_train_val']:
        data['train_x'] = np.vstack([data['train_x'], data['valid_x']])
        data['train_y'] = np.hstack([data['train_y'], data['valid_y']])
    
    # Summary writers
    tcost_ss = create_scalar_summary('training_cost')
    vcost_ss = create_scalar_summary('validation_cost')
    vacc_ss = create_scalar_summary('validation_accuracy')
    lr_ss = create_scalar_summary('learning_rate')
    
    # Configure tensorflow session
    config = config_init()
    if n_GPUs == 1:
        config.inter_op_parallelism_threads = 1 #prevent inter-session threads?
    sess = tf.Session(config=config)
    summary = tf.train.SummaryWriter(opt['log_path'], sess.graph)
    print('Summaries constructed...')
    
    sess.run(init)
    saver = tf.train.Saver()
    start = time.time()
    lr_ = opt['lr']
    epoch = 0
    step = 0.
    counter = 0
    best = 0.
    bs = opt['batch_size']
    print('Starting training loop...')
    while epoch < opt['n_epochs']:
        # Need batch_size*n_GPUs amount of data
        cost_total, acc_total, step = loop('train', sess, io, opt, data, loss,
                                           accuracy, lr, lr_, pt, optim=train_op,
                                           step=step)
        
        vloss_total, vacc_total, __ = loop('valid', sess, io, opt, data,
                                           loss, accuracy, lr, lr_,
                                           pt, optim=train_op)
        
        fd = {tcost_ss[0] : cost_total, vcost_ss[0] : vloss_total,
              vacc_ss[0] : vacc_total, lr_ss[0] : lr_}
        summaries = sess.run([tcost_ss[1], vcost_ss[1], vacc_ss[1], lr_ss[1]],
            feed_dict=fd)
        for summ in summaries:
            summary.add_summary(summ, step)
        best, counter, lr_ = get_learning_rate(opt, vacc_total, best, counter, lr_)
        print "[" + str(opt['trial_num']),str(epoch) + \
        "] Time: " + \
        "{:.3f}".format(time.time()-start) + ", Counter: " + \
        "{:d}".format(counter) + ", Loss: " + \
        "{:.5f}".format(cost_total) + ", Val loss: " + \
        "{:.5f}".format(vloss_total) + ", Train Acc: " + \
        "{:.5f}".format(acc_total) + ", Val acc: " + \
        "{:.5f}".format(vacc_total)
    
        epoch += 1
    
        if (epoch) % opt['save_step'] == 0:
            save_path = saver.save(sess, opt['checkpoint_path'])
            print("Model saved in file: %s" % save_path)
            
    # Save model and exit
    save_path = saver.save(sess, opt['checkpoint_path'])
    print("Model saved in file: %s" % save_path)
    sess.close()
    return tacc_total

def load_dataset(dir_name, subdir_name, prepend=''):
	"""Load dataset from subdirectory"""
	data_dir = dir_name + '/' + subdir_name
	data = {}
	data['train_x'] = np.load(data_dir + '/' + prepend + 'trainX.npy')
	data['train_y'] = np.load(data_dir + '/' + prepend + 'trainY.npy')[:,np.newaxis]
	data['valid_x'] = np.load(data_dir + '/' + prepend + 'validX.npy')
	data['valid_y'] = np.load(data_dir + '/' + prepend + 'validY.npy')[:,np.newaxis]
	return data

def create_scalar_summary(name):
	"""Create a scalar summary placeholder and op"""
	ss = []
	ss.append(tf.placeholder(tf.float32, [], name=name))
	ss.append(tf.scalar_summary(name+'_summary', ss[0]))
	return ss

def config_init():
	"""Default config settings"""
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = False
	return config

##### MAIN SCRIPT #####
def run(opt):
    # Parameters
    tf.reset_default_graph()
    
    # Default configuration
    opt['data_dir'] = '/home/daniel/data'
    opt['trial_num'] = 'A'
    opt['combine_train_val'] = False	
    
    data = load_dataset(opt['data_dir'], 'BSDS500_numpy', prepend='small_')
    opt['pos_weight'] = 100
    opt['model'] = getattr(equivariant, 'deep_bsd')
    opt['is_bsd'] = True
    opt['lr'] = 1e-1
    opt['batch_size'] = 4
    opt['std_mult'] = 1
    opt['momentum'] = 0.95
    opt['psi_preconditioner'] = 3.4
    opt['delay'] = 8
    opt['display_step'] = 10
    opt['save_step'] = 10
    opt['is_classification'] = True
    opt['n_epochs'] = 250
    opt['dim'] = 127
    opt['dim2'] = 160
    opt['n_channels'] = 3
    opt['n_classes'] = 2
    opt['n_filters'] = 32
    opt['filter_gain'] = 2
    opt['augment'] = False
    opt['lr_div'] = 10.
    opt['crop_shape'] = 0
    opt['log_path'] = './logs/deep_bsd'
    opt['checkpoint_path'] = './checkpoints/deep_bsd'
    
    # Check that save paths exist
    opt['log_path'] = opt['log_path'] + '/trial' + str(opt['trial_num'])
    opt['checkpoint_path'] = opt['checkpoint_path'] + '/trial' + \
                            str(opt['trial_num']) 
    if not os.path.exists(opt['log_path']):
        print('Creating log path')
        os.mkdir(opt['log_path'])
    if not os.path.exists(opt['checkpoint_path']):
        print('Creating checkpoint path')
        os.mkdir(opt['checkpoint_path'])
    opt['checkpoint_path'] = opt['checkpoint_path'] + '/model.ckpt'
    
    # Print out options
    for key, val in opt.iteritems():
        print(key + ': ' + str(val))
    return train_model(opt, data)


if __name__ == '__main__':
	deviceIdxs = [int(x.strip()) for x in sys.argv[1].split(',')]
	opt = {}
	opt['deviceIdxs'] = deviceIdxs

	run(opt)
	print("ALL FINISHED! :)")
