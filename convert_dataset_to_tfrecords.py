from __future__ import division

import os
import numpy as np

import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_write(X, Y, writer, tf_stuff):
    X = np.round(X * 255)
    #if len(X.shape) < 3:
    #    X = np.expand_dims(X, axis=2)
    x_serialised = tf_stuff['tf_sess'].run(tf_stuff['img_serialised'], feed_dict={
        tf_stuff['tf_input'] : X.astype(np.uint8),
    })
    #x_serialised = X.astype(np.uint8).tostring()
    y_serialised = Y.astype(np.int64).tostring()
    #currently, we require a 3d shape for both X and Y
    #so we add singleton dimensions as necessary 
    x_shape = []
    y_shape = []
    for i in xrange(3):
        if len(X.shape) <= i:
            x_shape.append(1)
        else:
            x_shape.append(X.shape[i])
        if len(Y.shape) <= i:
            y_shape.append(1)
        else:
            y_shape.append(Y.shape[i])
    x_shape_serialised = np.asarray(x_shape).astype(np.int64).tostring()
    y_shape_serialised = np.asarray(y_shape).astype(np.int64).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
    'x_raw': _bytes_feature(x_serialised),
    'y_raw': _bytes_feature(y_serialised),
    'x_shape': _bytes_feature(x_shape_serialised),
    'y_shape': _bytes_feature(y_shape_serialised),}))

    writer.write(example.SerializeToString())

def write_all_to_record(X, Y, writer):
    num_examples = X.shape[0]
    for i in xrange(num_examples):
        convert_write(X[i, :], Y[i], writer)
    writer.close()

def write_all_to_records(X, Y, writer_base_name, max_items_in_record, shape):
    tf_stuff = {}
    tf_stuff['tf_sess'] = tf.Session()
    tf_stuff['tf_input'] = tf.placeholder(tf.uint8)
    tf_stuff['img_serialised'] = tf.image.encode_jpeg(tf_stuff['tf_input'], optimize_size=True, quality=100)
    num_examples = X.shape[0]
    current_writer_idx = 0
    num_examples_in_current_writer = 0
    current_writer = tf.python_io.TFRecordWriter(writer_base_name + '_' + str(current_writer_idx) + '.tfrecords')
    perm = np.random.permutation(num_examples)
    for i in xrange(num_examples):
        if num_examples_in_current_writer >= max_items_in_record:
            print('Num examples written in file: ' + str(num_examples_in_current_writer))
            current_writer.close()
            current_writer_idx += 1
            current_writer = tf.python_io.TFRecordWriter(writer_base_name + '_' + str(current_writer_idx) + '.tfrecords')
            num_examples_in_current_writer = 0
            print('Started Writer: ' + str(current_writer_idx))
        #otherwise, read image an write to dataset
        idx = perm[i]
        convert_write(X[idx, :].reshape(shape), Y[idx], current_writer, tf_stuff)
        num_examples_in_current_writer += 1
    current_writer.close()
    print('Created ' + str(current_writer_idx + 1) + ' tfrecord files.')

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

#writer_train = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/cifar_numpy/train.tfrecords')
#writer_valid = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/cifar_numpy/valid.tfrecords')
#writer_test = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/cifar_numpy/test.tfrecords')
#write_all_to_record(data['train_x'], data['train_y'], writer_train)
#write_all_to_record(data['valid_x'], data['valid_y'], writer_valid)
#write_all_to_record(data['test_x'], data['test_y'], writer_test)

writer_train = '/home/sgarbin/TFR_CONVERSION/cifar_numpy/tfrecords/train'
writer_valid = '/home/sgarbin/TFR_CONVERSION/cifar_numpy/tfrecords/valid'
writer_test = '/home/sgarbin/TFR_CONVERSION/cifar_numpy/tfrecords/test'
write_all_to_records(data['train_x'], data['train_y'], writer_train, 2000, (32, 32, 3))
write_all_to_records(data['valid_x'], data['valid_y'], writer_valid, 2000, (32, 32, 3))
write_all_to_records(data['test_x'], data['test_y'], writer_test, 2000, (32, 32, 3))

print('Processing  rotated MNIST')
data = load_mnist_dataset('/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/')
#writer_train = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/train.tfrecords')
#writer_valid = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/valid.tfrecords')
#writer_test = tf.python_io.TFRecordWriter('/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/test.tfrecords')
#write_all_to_record(data['train_x'], data['train_y'], writer_train)
#write_all_to_record(data['valid_x'], data['valid_y'], writer_valid)
#write_all_to_record(data['test_x'], data['test_y'], writer_test)

writer_train = '/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/tfrecords/train'
writer_valid = '/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/tfrecords/valid'
writer_test = '/home/sgarbin/TFR_CONVERSION/mnist_rotation_new/tfrecords/test'
write_all_to_records(data['train_x'], data['train_y'], writer_train, 2000,(28, 28, 1))
write_all_to_records(data['valid_x'], data['valid_y'], writer_valid, 2000, (28, 28, 1))
write_all_to_records(data['test_x'], data['test_y'], writer_test, 2000, (28, 28, 1))

print(data['valid_y'][0:200])
