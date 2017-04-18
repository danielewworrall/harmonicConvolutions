'''Dirty hyperopt'''

import os
import sys
import time

import numpy as np

from run_mnist import main

"""Run the model for about 10 epochs and choose the one with the
hyperparameters, which minimize the objective the fastest
"""

def looper(n_trials):
   opt = {}
   
   opt['data_dir'] = './data'
   opt['aug_crop'] = 0
   opt['n_epochs'] = 200
   opt['filter_gain'] = 2
   opt['is_classification'] = True
   opt['combine_train_val'] = False
   opt['dim'] = 28
   opt['crop_shape'] = 0
   opt['n_channels'] = 1
   opt['n_classes'] = 10
   
   opt['load_settings'] = False
   
   best = -1e6
   best_opt = {}
   for i in xrange(n_trials):
      opt['learning_rate'] = log_rand(1e-3, 1e-2)
      
      opt['batch_size'] = int(rand(32,64))
      opt['display_step'] = 10000/opt['batch_size']
      opt['std_mult'] = rand(0.2, 1.)
      opt['psi_preconditioner'] = rand(1.,10.)
      opt['lr_div'] = rand(5., 15.)
      opt['delay'] = int(rand(8,12))

      opt['n_filters'] = 8
      opt['filter_size'] = int(rand(2,6))
      opt['n_rings'] = int(rand(1,5))
      
      val_acc = main(opt)
      
      if val_acc > best:
         best_opt = opt
         best = np.maximum(best, val_acc)
      
      print
      print('Best validation acc. so far: {:f}'.format(best))
      for key, val in best_opt.iteritems():
         print(key, val)
      print
   
   print('Best validation acc.: {:f}'.format(best))
   for key, val in best_opt.iteritems():
      print(key, val)


def rand(low, high):
   return low + (high-low)*np.random.rand()


def log_rand(low, high):
   log_low = np.log10(low)
   log_high = np.log10(high)
   return np.power(10., log_low + (log_high-log_low)*np.random.rand())


if __name__ == '__main__':
   n_trials = 16
   looper(n_trials)
