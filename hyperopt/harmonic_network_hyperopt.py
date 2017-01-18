'''Hyperopt CIFAR'''

import os
import sys
import time
sys.path.append('../')

import cPickle as pkl
import numpy as np
import tensorflow as tf

from train import create_opt_data, build_all_and_train


def main(opt, data, n):
    """Train the networks"""
    for i in xrange(n):
        opt['use_io_queues'] = False
        opt['aug_crop'] = np.random.randint(5)
        opt['n_epochs'] = 50 + np.random.randint(200)
        opt['batch_size'] = int(log_uniform(2,3,3))
        opt['lr']  = log_uniform(10,-4,3)
        opt['optimizer'] = tf.train.AdamOptimizer
        opt['std_mult'] = 0.3+0.2*np.random.rand()
        opt['delay'] = 8
        opt['psi_preconditioner'] = 4.
        opt['momentum'] = 0.85+0.1*np.random.rand()
        opt['log_path'] = '../logs/deep_cifar' + str(n)
        opt['checkpoint_path'] = '../checkpoints/deep_cifar' + str(n)
        #build the model and train it
        opt['res'] = build_all_and_train(opt, data)
        fname = './results/trial'+str(i)+'.pkl'
        with open(fname, 'w') as fp:
            pkl.dump(opt, fp, protocol=pkl.HIGHEST_PROTOCOL)
    print("ALL FINISHED! :)")

def log_uniform(base, logmin_, logmax_):
    """Return log uniform samples"""
    return np.power(base,logmin_+logmax_*np.random.rand())

if __name__ == '__main__':
    opt = {}
    opt['deviceIdxs'] = [0,]
    opt['datasetIdx'] = 'cifar10'
    opt['model'] = 'deep_cifar'
    opt['data_dir'] = 'cifar10'
    #create configuration for different tests
    amended_opt, data = create_opt_data(opt)
    main(amended_opt, data, 10)
