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
        self.__maybe_create('num_threads_per_queue', 1)
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
            print('NOTE: Option [' + key + '] is specified by user. Not using default.')
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

    def set_option(key, value):
        self.__set(key, value)
    
    def set_data_option(key, value):
        self.__data_set(key, value)

    def create_options(self):
        # Default configuration
        self.__maybe_create('save_step', 10)
        self.__maybe_create('trial_num', 'A')
        self.__maybe_create('lr_div', 10)
        self.__maybe_create('augment', False)
        self.__maybe_create('is_bsd', False)
        self.__maybe_create('train_data_fraction', 1.0)
        #now create options specific to datasets
        if self.__get('dataset') == 'rotated_mnist':
            self.__create_options_rotated_mnist()
        elif self.__get('dataset') == 'cifar10':
            self.__create_options_cifar10()
        else:
            print('ERROR: not implemented')
            return False
        return True


    def __create_options_rotated_mnist(self):
        #setup data feeding
        mnist_dir = self.__get('data_dir') + '/mnist_rotation_new'
        #data feeding choice
        self.__set('use_io_queues', True)
        if self.__get('use_io_queues'):
            #we can use this convenience function to get all the data
            data = discover_and_setup_tfrecords(mnist_dir, 
                self.data, use_train_fraction = self.__get('train_data_fraction'))
            #define the types stored in the .tfrecords files
            self.__data_set('x_type', tf.uint8)
            self.__data_set('y_type', tf.int64)
            #let's define some functions to reshape data
            #note: [] means nothing will happen
            self.__data_set('x_target_shape', [28, 28, 1, 1, 1])
            self.__data_set('y_target_shape', [1]) #a 'squeeze' is automatically applied here
            #set the data decoding function
            self.__data_set('data_decode_function', \
                (lambda features : [tf.image.decode_jpeg(features['x_raw']), \
                    tf.decode_raw(features['y_raw'], data['y_type'], name="decodeY")]))
            #set the data processing function
            self.__data_set('data_process_function', \
                (lambda x, y : [tf.image.per_image_standardization(tf.cast(x, tf.float32)), y]))
        self.__maybe_create('aug_crop', 0) #'crop margin'
        self.__maybe_create('n_epochs', 200)
        self.__maybe_create('batch_size', 46)
        self.__maybe_create('lr', 0.0076)
        self.__maybe_create('optimizer', tf.train.AdamOptimizer)
        self.__maybe_create('momentum', 0.93)
        self.__maybe_create('std_mult', 0.7)
        self.__maybe_create('delay', 12)
        self.__maybe_create('psi_preconditioner', 7.8)
        self.__maybe_create('filter_gain', 2)
        self.__maybe_create('filter_size', 3)
        self.__maybe_create('n_filters', 8)
        self.__maybe_create('display_step', 10000/self.__get('batch_size')*3.)
        self.__maybe_create('is_classification', True)
        self.__maybe_create('combine_train_val', False)
        self.__maybe_create('dim', 28)
        self.__maybe_create('crop_shape', 0)
        self.__maybe_create('n_channels', 1)
        self.__maybe_create('n_classes', 10)
        self.__maybe_create('log_path', './logs/deep_mnist/trialA')
        self.__maybe_create('checkpoint_path', './checkpoints/deep_mnist/trialA')
        
    def __create_options_cifar10(self):
        #setup data feeding
        mnist_dir = self.__get('data_dir') + '/cifar10'
        #data feeding choice
        self.__set('use_io_queues', True)
        if self.__get('use_io_queues'):
            #we can use this convenience function to get all the data
            data = discover_and_setup_tfrecords(mnist_dir, 
                self.data, use_train_fraction = self.__get('train_data_fraction'))
            #define the types stored in the .tfrecords files
            self.__data_set('x_type', tf.uint8)
            self.__data_set('y_type', tf.int64)
            #let's define some functions to reshape data
            #note: [] means nothing will happen
            self.__data_set('x_target_shape', [32, 32, 3, 1, 1])
            self.__data_set('y_target_shape', [1]) #a 'squeeze' is automatically applied here
            #set the data decoding function
            self.__data_set('data_decode_function', \
                (lambda features : [tf.image.decode_jpeg(features['x_raw']), \
                    tf.decode_raw(features['y_raw'], data['y_type'], name="decodeY")]))
            #set the data processing function
            self.__data_set('data_process_function', \
                (lambda x, y : [tf.image.per_image_standardization(tf.cast(x, tf.float32)), y]))
        self.__maybe_create('is_classification', True)
        self.__maybe_create('dim', 32)
        self.__maybe_create('crop_shape', 0)
        self.__maybe_create('aug_crop', 3)
        self.__maybe_create('n_channels', 3)
        self.__maybe_create('n_classes', 10)
        self.__maybe_create('n_epochs', 250)
        self.__maybe_create('batch_size', 32)
        self.__maybe_create('lr', 0.01)
        self.__maybe_create('optimizer', tf.train.AdamOptimizer)
        self.__maybe_create('std_mult', 0.4)
        self.__maybe_create('delay', 8)
        self.__maybe_create('psi_preconditioner', 7.8)
        self.__maybe_create('filter_gain', 2)
        self.__maybe_create('filter_size', 3)
        self.__maybe_create('n_filters', 4*10)	# Wide ResNet
        self.__maybe_create('resnet_block_multiplicity', 3)
        self.__maybe_create('augment', True)
        self.__maybe_create('momentum', 0.93)
        self.__maybe_create('display_step', 25)
        self.__maybe_create('is_classification', True)
        self.__maybe_create('n_channels', 3)
        self.__maybe_create('n_classes', 10)
        self.__maybe_create('log_path', './logs/deep_cifar')
        self.__maybe_create('checkpoint_path', './checkpoints/deep_cifar')