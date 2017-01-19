import sys
import os
import numpy as np
import tensorflow as tf

from io_helpers import load_dataset, download_dataset, discover_and_setup_tfrecords
import harmonic_network_models

class settings():
    def __init__(self, opt):
        self.opt = opt
        self.data = {}
        #check that we have all the required options
        if 'deviceIdxs' in self.opt and \
            'dataset' in self.opt and \
            'model' in self.opt and \
            'data_dir' in self.opt:
            #replace model name with function 
            opt['model'] = getattr(harmonic_network_models, opt['model'])
            return
        else:
            print('ERROR: You must specify the following')
            print('\t deviceIdxs')
            print('\t dataset')
            print('\t model')
            print('\t data_dir')

    def __maybe_create(self, key, value):
        if key in self.opt:
            return
        else:
            self.opt[key] = value

    def __get(self, key):
        return self.opt[key]

    def __set(self, key, value):
        self.opt[key] = value

    def __data_get(self, key):
        return self.data[key]

    def __data_set(self, key, value):
        self.data[key] = value
    
    def get_options(self):
        return self.opt

    def get_data_options(self):
        return self.data

    def create_options(self):
        # Default configuration
        self.__maybe_create('save_step', 10)
        self.__maybe_create('display_step', 1e6)
        self.__maybe_create('lr', 3e-2)
        self.__maybe_create('batch_size', 50)
        self.__maybe_create('n_epochs', 100)
        self.__maybe_create('n_filters', 8)
        self.__maybe_create('trial_num', 'A')
        self.__maybe_create('combine_train_val', False)
        self.__maybe_create('std_mult', 0.3)
        self.__maybe_create('filter_gain', 2)
        self.__maybe_create('momentum', 0.93)
        self.__maybe_create('psi_preconditioner', 3.4)
        self.__maybe_create('delay', 8)
        self.__maybe_create('lr_div', 10)
        self.__maybe_create('augment', False)
        self.__maybe_create('crop_shape', 0)
        self.__maybe_create('log_path', 'logs/current')
        self.__maybe_create('checkpoint_path', 'checkpoints/current')
        self.__maybe_create('is_bsd', False)
        #now create options specific to datasets
        if self.__get('dataset') == 'rotated_mnist':
            self.__create_options_rotated_mnist()
        else:
            print('ERROR: no implemented')
            return False
        return True


    def __create_options_rotated_mnist(self):
        # Download MNIST if it doesn't exist
        if not os.path.exists(self.__get('data_dir') + '/mnist_rotation_new'):
            download_dataset(opt)
        # Load dataset
        mnist_dir = self.__get('data_dir') + '/mnist_rotation_new'
        #data feeding choice
        self.__set('use_io_queues', True)
        if self.__get('use_io_queues'):
            #we can use this convenience function to get all the data
            data = discover_and_setup_tfrecords(mnist_dir, self.data, use_train_fraction=0.5)
            #let's define some functions to reshape data
            #note: [] means nothing will happen
            self.__data_set('x_target_shape', [28, 28, 1, 1, 1])
            self.__data_set('y_target_shape', [1]) #a 'squeeze' is automatically applied here
            #define the types stored in the .tfrecords files
            self.__data_set('x_type', tf.uint8)
            self.__data_set('y_type', tf.int64)
            #set the data decoding function
            self.__data_set('data_decode_function', \
                (lambda features : [tf.image.decode_jpeg(features['x_raw']), \
                    tf.decode_raw(features['y_raw'], data['y_type'], name="decodeY")]))
            #set the data processing function
            self.__data_set('data_process_function', \
                (lambda x, y : [tf.image.per_image_standardization(tf.cast(x, tf.float32)), y]))
        self.__set('aug_crop', 0) #'crop margin'
        self.__set('n_epochs', 200)
        self.__set('batch_size', 46)
        self.__set('lr', 0.0076)
        self.__set('optimizer', tf.train.AdamOptimizer)
        self.__set('momentum', 0.93)
        self.__set('std_mult', 0.7)
        self.__set('delay', 12)
        self.__set('psi_preconditioner', 7.8)
        self.__set('filter_gain', 2)
        self.__set('filter_size', 3)
        self.__set('n_filters', 8)
        self.__set('display_step', 10000/self.__get('batch_size')*3.)
        self.__set('is_classification', True)
        self.__set('combine_train_val', False)
        self.__set('dim', 28)
        self.__set('crop_shape', 0)
        self.__set('n_channels', 1)
        self.__set('n_classes', 10)
        self.__set('log_path', './logs/deep_mnist/trialA')
        self.__set('checkpoint_path', './checkpoints/deep_mnist/trialA')