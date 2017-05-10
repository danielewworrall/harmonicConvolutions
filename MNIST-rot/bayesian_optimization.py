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
   dimensions = [(25, 75),       # batch size
                 (1e-4, 1e-1),  # learning rate
                 (0.1, 1.),     # std mult
                 (3, 6),        # filter_size
                 (2, 5)]        # n_rings

   x0 = [46, 0.0076, 0.7, 5, 4]

   print(gp_minimize(wrapper_function, dimensions, x0=x0, n_calls=n_trials, verbose=True, callback=dump))


def wrapper_function(dimension):
   """Wrapper around the neural network training function"""

   parser = argparse.ArgumentParser()
   parser.add_argument("--mode", help="model to run {hnet,baseline}", default="hnet")
   parser.add_argument("--save_name", help="name of the checkpoint path", default="my_model")
   parser.add_argument("--data_dir", help="data directory", default='./data')
   parser.add_argument("--default_settings", help="use default settings", type=bool, default=False)
   parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool, default=False)
   parser.add_argument("--delete_existing", help="delete the existing auxilliary files", type=bool, default=True)
   args = parser.parse_args()

   # Default configuration
   args.n_epochs = 250
   args.filter_gain = 2
   args.n_filters = 8
   args.save_step = 5
   args.height = 28
   args.width = 28

   args.n_channels = 1
   args.n_classes = 10
   args.lr_div = 10.
   args.augment = True
   args.sparsity = True

   args.test_path = args.save_name
   args.log_path = os.path.join('./logs/', args.test_path)
   args.checkpoint_path = os.path.join('./checkpoints/', args.test_path)
   args.test_path = os.path.join('./', args.test_path)

   args.batch_size = dimension[0]
   args.learning_rate = dimension[1]
   args.std_mult = dimension[2]
   args.filter_size = dimension[3]
   args.n_rings = dimension[4]


   for arg in vars(args):
      print(arg, getattr(args, arg))
   print

   valid_acc = main(args)
   return 1-valid_acc


if __name__ == '__main__':
   optimize(30)
