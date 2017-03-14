"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import os
import numpy as np
import binvox_rw
import glob
import multiprocessing

# save our result
def save_binvox(inp, prefix='model_'):
  for i in range(inp.shape[0]):
    cur_vol = inp[i,:,:,:,0]
    cur_vol = np.transpose(cur_vol, [2,1,0])
    cur_vol = cur_vol>0.5 # binary
    #with open('data/model.binvox', 'rb') as f:
    #  model = binvox_rw.read_as_3d_array(f)
    cur_model = binvox_rw.Voxels(
            data=cur_vol, 
            dims=list(cur_vol.shape), 
            translate=[0,0,0], 
            scale=1.0, 
            axis_order='xyz')
    with open(prefix + str(i) + '.binvox', 'wb') as f:
      cur_model.write(f)


def read_binbox(path):
  with open(path, 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)
    result = model.data.copy().astype(np.int8)
    result = np.transpose(result, [2,1,0])
    return result
  return None

 
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):
  def __init__(self, file_list, label_list, one_hot):
    assert len(file_list) == len(label_list), (
        "files: %s labels: %s" % (len(file_list), len(label_list)))
    self._num_examples = len(file_list)
    # read first file to get width, height, depth
    def get_dims(fname):
        vol = read_binbox(fname)
        return vol.shape[0], vol.shape[1], vol.shape[2]
    
    width, height, depth = get_dims(file_list[0])

    self._volumes = np.empty([self._num_examples, depth, height, width, 1], dtype=np.int8)

    p = multiprocessing.Pool(processes=4)
    for i in range(self._num_examples):
        fname = file_list[i]
        cur_vol = p.apply_async(read_binbox, [fname])
        self._volumes[i,:,:,:,0] = cur_vol.get()
        #cur_vol = read_binbox(fname)
        #self._volumes[i,:,:,:,0] = cur_vol
    p.close()
    p.join()

    self.perm = np.arange(self._num_examples, dtype=np.int64)
    self._labels = np.reshape(np.array(label_list), [-1])
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
  @property
  def volumes(self):
    return self._volumes
  
  @property
  def labels(self):
    return self._labels
  
  @property
  def num_examples(self):
    return self._num_examples

  def num_steps(self, batch_size):
    return int(self._num_examples/batch_size)
  
  @property
  def epochs_completed(self):
    return self._epochs_completed
  
  def shuffle(self):
      np.random.shuffle(self.perm)

  def next_batch(self, batch_size, doperm=True):
      if doperm and self._index_in_epoch==0:
	self.shuffle()

      """Return the next `batch_size` examples from this data set."""
      start = self._index_in_epoch
      self._index_in_epoch += batch_size
      if self._index_in_epoch > self._num_examples:
        # Shuffle the data
        self.shuffle()

        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_examples
      end = self._index_in_epoch

      inds = np.arange(start, end, dtype=np.int64)
      if doperm:
          inds = self.perm[inds]
      volumes = self._volumes[self.perm[start:end], :,:,:,:]
      volumes = volumes.astype(np.float32)
      labels = self._labels[self.perm[start:end]]
      return volumes, labels

  
def read_data_sets(path, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  #print(path)
  #print(os.path.expanduser(path))
  #print(os.path.realpath(os.path.expanduser(path)))
  path = os.path.realpath(os.path.expanduser(path))
  print('Reading', path)

  # TODO
  #train_split = 0.8
  #validation_split = 0.1
  #test_split = 0.1
  train_split = 0.5
  validation_split = 0.01
  test_split = 0.01

  class_folders = sorted(glob.glob(os.path.join(path, '*')))
  classes = [os.path.basename(os.path.normpath(class_folder)) for class_folder in class_folders]
  #print(classes)
  train_file_list = []
  validation_file_list = []
  test_file_list = []
  train_labels = []
  validation_labels = []
  test_labels = []

  all_file_count = 0
  class_folders = class_folders[3:4]
  print(class_folders)
  for i in range(len(class_folders)):
    file_list = sorted(glob.glob(os.path.join(class_folders[i], '*/model.binvox')))
    all_file_count += len(file_list)

    rem_file_size = len(file_list)
    train_split_size = np.min([rem_file_size, np.ceil(len(file_list)*train_split).astype(np.int)])
    rem_file_size = np.max([0, len(file_list) - train_split_size])
    validation_split_size = np.ceil(len(file_list)*validation_split).astype(np.int)
    rem_file_size = np.max([0, len(file_list) - train_split_size - validation_split_size])
    test_split_size = np.min([rem_file_size, np.ceil(len(file_list)*test_split).astype(np.int)])

    cur_train_file_list = file_list[0:train_split_size]
    cur_validation_file_list = file_list[train_split_size: (train_split_size + validation_split_size)]
    cur_test_file_list = file_list[(train_split_size + validation_split_size):(train_split_size + validation_split_size + test_split_size)]

    cur_train_labels = [i]*train_split_size # DANGER BEWARE do not change values in this list
    cur_validation_labels = [i]*validation_split_size # DANGER BEWARE do not change values in this list
    cur_test_labels = [i]*test_split_size # DANGER BEWARE do not change values in this list

    #print(os.path.basename(os.path.normpath(os.path.dirname(file_list[0]))))

    train_file_list.extend(cur_train_file_list)
    validation_file_list.extend(cur_validation_file_list)
    test_file_list.extend(cur_test_file_list)
    train_labels.extend(cur_train_labels)
    validation_labels.extend(cur_validation_labels)
    test_labels.extend(cur_test_labels)

  print('Files to read:', all_file_count)

  data_sets.train = DataSet(train_file_list, train_labels, one_hot)
  data_sets.validation = DataSet(validation_file_list, validation_labels, one_hot)
  data_sets.test = DataSet(test_file_list, test_labels, one_hot)
  return data_sets


def test():
  dataset = read_data_sets('~/scratch/Datasets/ShapeNetVox32')
  tmp1, tmp2 = dataset.train.next_batch(2)
  print(dataset.train.volumes.shape)
  print(tmp1.shape)
  print(np.amax(tmp1))
  print(np.amin(tmp1))
  print(tmp2)

def main(argv=None):  # pylint: disable=unused-argument
  test()

if __name__ == '__main__':
  main() 
