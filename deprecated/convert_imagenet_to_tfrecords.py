from __future__ import division

import timeit
import os
import pickle
import numpy as np
import scipy as sp 
from scipy import misc

import OpenImageIO as oiio
from OpenImageIO import FLOAT, ImageInput

import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def load_image(filename):
	# Open the input file
	image = ImageInput.open(filename)
	# Compute the size
	spec = image.spec()
	# Create empty ndarray and set its memory
	pixels = np.empty((spec.height, spec.width, spec.nchannels), dtype=np.float32)
	pixels.data = image.read_image(FLOAT) #fill existing ndarray
	return pixels

def convert_write(X, Y, writer, tf_stuff):
    #make grey-scale images colour
    if len(X.shape) < 3:
        X = np.stack([X, X, X], axis=2)
    
    x_serialised = tf_stuff['tf_sess'].run(tf_stuff['img_serialised'], feed_dict={
        tf_stuff['tf_input'] : X.astype(np.uint8),
    })
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

def read_train_file(file, new_data_directory):
    #as in http://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list-with-python
    with open(file) as f:
        files = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    files = [x.strip() for x in files]
    files.sort()
    print('Found ' + str(len(files)) + ' training image files.')
    labels = []
    files = [i.replace('/media/daniel/DATA/ImageNet', '', 1) for i in files]
    #now get labels
    label_map = {}
    category_counter = 0
    for i in files:
        stripped = i.replace('/ILSVRC2012_img_train/', '', 1)
        category = stripped.split('/', 1)[0]
        if not category in label_map:
            label_map[category] = category_counter
            category_counter += 1
        labels.append(label_map[category])
    print('Found ' + str(category_counter) + ' categories.')
    #make files into full paths
    files = [new_data_directory + '' + i for i in files]
    return files, labels, label_map

def read_valid_file(file, new_data_directory, label_map):
    with open(file) as f:
        entries = f.readlines()
    files = []
    labels = []
    for e in entries:
        image_name, category = e.split(' ')
        image_name = image_name.strip()
        category = category.strip()
        files.append(new_data_directory + '/' + image_name)
        if category in label_map:
            labels.append(label_map[category])
        else:
            print('ERROR: validation category not processed in train file!')
    print('Found ' + str(len(files)) + ' validation examples.')       
    return files, labels

def create_label_category_mapping_file(category_map, mapping_file, target_file):
    with open(mapping_file) as f:
        entries = f.readlines()
    label_mappings = {}
    found_categories = 0
    #associate each category with a numeric class and then with a text description
    for e in entries:
        label, text = e.split('\t')
        label = label.strip()
        if label in label_map:
            label_mappings[category_map[label]] = [text, label]
            found_categories += 1
        #else:
            #print('ERROR: category [' + label + '] not found in mapping file!')
    print(str(found_categories) + ' categories correctly mapped in dictionary')
    #save dictionary with pkl
    with open(target_file, 'wb') as file:
        pickle.dump(label_mappings, file, pickle.HIGHEST_PROTOCOL)
    return label_mappings

def process_image_list(image_list, labels, writer_max_size, max_items_in_record, writer_base_name):
    tf_stuff = {}
    tf_stuff['tf_sess'] = tf.Session()
    tf_stuff['tf_input'] = tf.placeholder(tf.uint8)
    tf_stuff['img_serialised'] = tf.image.encode_jpeg(tf_stuff['tf_input'], optimize_size=True, quality=95)
    current_writer_size = 0
    current_writer_idx = 0
    num_examples_in_current_writer = 0
    current_writer = tf.python_io.TFRecordWriter(writer_base_name + '_' + str(current_writer_idx) + '.tfrecords')
    num_examples = len(image_list)
    perm = np.random.permutation(num_examples)
    start = timeit.timeit()
    num_errors = 0
    num_grey_scale = 0
    for i in xrange(num_examples):
        #if num_errors > 5:
        #    break
        idx = perm[i]
        image_name = image_list[idx]
        label = labels[idx]
        #make sure we only write files of a certain size
        if num_examples_in_current_writer >= max_items_in_record:
            print('Num examples written in file: ' + str(num_examples_in_current_writer))
            current_writer.close()
            end = timeit.timeit()
            print('Elapsed Time: ' + str(end - start))
            current_writer_idx += 1
            current_writer = tf.python_io.TFRecordWriter(writer_base_name + '_' + str(current_writer_idx) + '.tfrecords')
            current_writer_size = 0
            num_examples_in_current_writer = 0
            start = timeit.timeit()
            print('Started Writer: ' + str(current_writer_idx))
        #otherwise, read image an write to dataset
        try:
            image = sp.misc.imread(image_name)
            if len(image.shape) < 3:
                num_grey_scale += 1
            #image = load_image(image_name)
            np_label = np.zeros(1, dtype=np.int64)
            np_label[0] = label
            convert_write(image, np_label, current_writer, tf_stuff)
        except:
            print('Could not read file: ' + image_name)
            num_errors += 1
        #current_writer_size += image.nbytes + 48 #for 2 64bit 3-shape size tensors
        num_examples_in_current_writer += 1
    current_writer.close()
    print('Created ' + str(current_writer_idx + 1) + ' tfrecord files.')
    print('Num Errors: ' + str(num_errors))
    print('Num Grey-scale conversions: ' + str(num_grey_scale))
    return

#process CIFAR
print('Processing ImageNet  2012')

#get list of training images and labels
train_files, train_labels, label_map = read_train_file('/home/sgarbin/TFR_CONVERSION/train.txt',
    '/home/sgarbin/TFR_CONVERSION/imagenet')

#do the same for the validation set
valid_files, valid_labels = read_valid_file('/home/sgarbin/TFR_CONVERSION/validation.txt',
    '/home/sgarbin/TFR_CONVERSION/imagenet/ILSVRC2012_img_val', label_map)

#save the dictionary
create_label_category_mapping_file(label_map, '/home/sgarbin/TFR_CONVERSION/wnid.txt',
    '/home/sgarbin/TFR_CONVERSION/imagenet/tfrecord/dictionary.pkl')

#save the actual files
process_image_list(train_files, train_labels, 2e+9, 10000, '/home/sgarbin/TFR_CONVERSION/imagenet/tfrecord/train')
print('Finished train set, now processing validation images...')
process_image_list(valid_files, valid_labels, 2e+9, 10000, '/home/sgarbin/TFR_CONVERSION/imagenet/tfrecord/valid')