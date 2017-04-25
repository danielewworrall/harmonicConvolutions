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

from run_BSD import main
from skopt import gp_minimize


def dump(res):
   with open('results.pkl', 'w') as fp:
      pkl.dump(res.x, fp)


def optimize(n_trials):
   """Run the gp_minimize function"""
   dimensions = [(80, 90), # batch size
                 (200, 210), # image dims
                 (1e-10, 1e-8, 'log-uniform'), # learning rate
                 (1e-5, 1e-3, 'log-uniform'), # learning rate divisor
                 (8, 15)] # jitter

   x0 = [80, 202, 2e-10, 1e-5, 10]

   print gp_minimize(wrapper_function, dimensions, x0=x0, n_calls=100, verbose=True, callback=dump)


def wrapper_function(dimension):
   """Wrapper around the neural network training function"""

   parser = argparse.ArgumentParser()
   parser.add_argument("--n_channels", help="number of input image channels", type=int, default=3)
   parser.add_argument("--n_classes", help="number of output classes", type=int, default=2)
   parser.add_argument("--n_iterations", help="number of minibatches to pass", type=int, default=3000)
   parser.add_argument("--learning_rate_step", help="interval to divide learning rate by 10", type=int, default=1000)
   parser.add_argument("--momentum", help="momentum rate for stochastic gradient descent", type=float, default=0.9)
   parser.add_argument("--pretrained_path", help="path to pretrained model checkpoint", default='./pretrained/resnet_v1_50.ckpt')
   parser.add_argument("--preprocess", help="whether to preprocess images", type=bool, default=True)
   parser.add_argument("--min_after_dequeue", help="minimum number of images to keep in RAM", type=int, default=1000)
   parser.add_argument("--train_file", help="location of training file", default='./sets/train_randomized/train_0_randomized.txt')
   parser.add_argument("--valid_file", help="location of training file", default='./sets/valid_randomized/valid_0_randomized.txt')
   parser.add_argument("--train_dir", help="directory of training examples", default="./data/")
   parser.add_argument("--valid_dir", help="directory of validation examples", default="./data/")
   parser.add_argument("--save_dir", help="directory to save results", default="./checkpoints")
   parser.add_argument("--log_dir", help="directory to save results", default="./logs")
   parser.add_argument("--save_interval", help="number of iterations between saving model", type=int, default=50)
   parser.add_argument("--delete_existing", help="delete existing models and logs in same folders", default=True)
   args = parser.parse_args()

   args.batch_size = dimension[0]
   args.height = dimension[1]
   args.width = args.height
   args.learning_rate = dimension[2]
   args.learning_rate_divisor = dimension[3]
   args.jitter = dimension[4]

   for arg in vars(args):
      print arg, getattr(args, arg)
   print

   valid_acc = main(args)
   return 1 - valid_acc


if __name__ == '__main__':
   optimize(20)
