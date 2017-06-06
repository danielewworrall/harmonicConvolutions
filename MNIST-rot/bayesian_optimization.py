"""Bayesian hyperparameter optimization (hyperopt) is very modern and useful
technique to optimize the hyperparameters of a non-differentiable loss function.
We make use of the scikit-optimize package, which contains a Gaussian Process
hyperopt method.
"""

import argparse
import cPickle as pkl
import os
import sys
import time

import numpy as np

from run_mnist import main
from skopt import gp_minimize


def dump(res):
   with open('results.pkl', 'w') as fp:
      pkl.dump(res.x, fp)


def optimize(n_trials):
   """Run the gp_minimize function"""
   dimensions = [(32, 64),       # batch size
                 (1e-3, 1e-1),   # learning rate
                 (0.5, 1.1),     # std mult
                 (3, 6),         # filter_size
                 (2, 5),         # n_rings
                 (1.,10.),       # phase_preconditioner
                 (150,300),      # n_epochs
                 (0.1,0.9)]      # bandwidth

   x0 = [46, 0.0076, 0.7, 5, 4, 7.8, 200, 0.3]

   print gp_minimize(wrapper_function, dimensions, x0=x0, n_calls=n_trials, verbose=True, callback=dump)


def wrapper_function(dimension):
   """Wrapper around the neural network training function"""

   parser = argparse.ArgumentParser()
   args = parser.parse_args()
   args.data_dir = "./data"
   args.default_settings = False 
   args.combine_train_val = True
   args.delete_existing = True

   # Default configuration
   args.filter_gain = 2
   args.n_filters = 7
   args.save_step = 5
   args.dim = 28

   args.n_channels = 1
   args.lr_div = 10.
   args.n_classes = 10

   args.batch_size = dimension[0]
   args.learning_rate = dimension[1]
   args.std_mult = dimension[2]
   args.filter_size = dimension[3]
   args.n_rings = dimension[4]
   args.phase_preconditioner = dimension[5]
   args.n_epochs = dimension[6]
   args.bw = dimension[7]

   for arg in vars(args):
      print arg, getattr(args, arg)
   print

   valid_acc = main(args)
   return 100.*(1.-valid_acc)


if __name__ == '__main__':
   optimize(100)
