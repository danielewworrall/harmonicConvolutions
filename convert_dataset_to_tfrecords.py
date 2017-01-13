from __future__ import division

import os
import numpy as np

import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int32_feature(value):
    return tf.train.Feature(int32_list=tf.train.Int32List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_write(X, Y, writer):
    x_serialised = X.astype(np.float32).tostring()
    y_serialised = Y.astype(np.float32).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
    'x_raw': _bytes_feature(x_serialised),
    'y_raw': _bytes_feature(y_serialised)}))

    writer.write(example.SerializeToString())

def write_all_to_record(X, Y, writer):
    num_examples = X.shape[0]
    for i in xrange(num_examples):
        convert_write(X[i, :], Y[i], writer)
    writer.close()

def load_dataset(dir_name, subdir_name=''):
    if subdir_name != '':
        data_dir = dir_name + '/' + subdir_name
    else:
        data_dir = dir_name
    print('Loading data from directory: [ ' + data_dir + ' ]...')

    data = {}
    data['train_x'] = np.load(data_dir + 'trainX.npy')
    data['train_y'] = np.load(data_dir + 'trainY.npy')
    data['valid_x'] = np.load(data_dir + 'validX.npy')
    data['valid_y'] = np.load(data_dir + 'validY.npy')
    data['test_x'] = np.load(data_dir + 'testX.npy')
    if os.path.exists(data_dir + 'testY.npy'):
        data['test_y'] = np.load(data_dir + 'testY.npy')
    return data

def load_mnist_dataset(dir_name, subdir_name=''):
    if subdir_name != '':
        data_dir = dir_name + '/' + subdir_name
    else:
        data_dir = dir_name
    print('Loading data from directory: [ ' + data_dir + ' ]...')
    
    train = np.load(data_dir + '/rotated_train.npz')
    valid = np.load(data_dir + '/rotated_valid.npz')
    test = np.load(data_dir + '/rotated_test.npz')
    data = {}
    data['train_x'] = train['x']
    data['train_y'] = train['y']
    data['valid_x'] = valid['x']
    data['valid_y'] = valid['y']
    data['test_x'] = test['x']
    data['test_y'] = test['y']
    return data

#process CIFAR
print('Processing CIFAR10')
data = load_dataset('/home/sgarbin/TFR_CONVERSION/cifar_numpy/')

writer_train = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/cifar_numpy/train.tfrecords')
writer_valid = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/cifar_numpy/valid.tfrecords')
writer_test = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/cifar_numpy/test.tfrecords')

write_all_to_record(data['train_x'], data['train_y'], writer_train)
write_all_to_record(data['valid_x'], data['valid_y'], writer_valid)
write_all_to_record(data['test_x'], data['test_y'], writer_test)

print('Processing  rotated MNIST')
data = load_mnist_dataset('/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/')
writer_train = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/train.tfrecords')
writer_valid = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/valid.tfrecords')
writer_test = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/test.tfrecords')

write_all_to_record(data['train_x'], data['train_y'], writer_train)
write_all_to_record(data['valid_x'], data['valid_y'], writer_valid)
write_all_to_record(data['test_x'], data['test_y'], writer_test)
