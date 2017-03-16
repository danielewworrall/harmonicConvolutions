"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import os
import numpy as np
from npyio import NpyTarReader

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class_names = ['airplane', 'bathtub', 'bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup', 'curtain', 'desk', 'door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person', 'piano', 'plant', 'radio','range_hood','sink','sofa', 'stairs', 'stool', 'table','tent','toilet','tv_stand', 'vase', 'wardrobe', 'xbox']
class_id = dict(zip(class_names, range(len(class_names))))

class DataSet(object):
  def __init__(self, path, one_hot):
    reader = NpyTarReader(path)
    self._num_examples = reader.num_files
    for ix, (x, name) in enumerate(reader):
        if ix==0:
            # read first file to get width, height, depth
            depth, height, width = x.shape # TODO not sure if order is correct
            self._volumes = np.empty([self._num_examples, depth, height, width, 1], dtype=np.uint8)
            self._labels = np.empty([self._num_examples], dtype=np.uint16)
        self._volumes[ix,:,:,:,0] = x
        name = name[4:]
        name = name[:name.find('0')-1]
        label = class_id[name]
        self._labels[ix] = label

    self.perm = np.arange(self._num_examples, dtype=np.int64)
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

def read_data_sets(basedir, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  basedir = os.path.realpath(os.path.expanduser(basedir))
  print('Reading', basedir)

  data_sets.train = DataSet(os.path.join(basedir, 'shapenet10_train_nr.tar'), one_hot)
  data_sets.test = DataSet(os.path.join(basedir, 'shapenet10_test_nr.tar'), one_hot)
  data_sets.validation = data_sets.test # no decision based on validation sets

  return data_sets
  
def test():
  dataset = read_data_sets('~/Documents/Datasets/ModelNet/')
  #dataset = read_data_sets('~/ShapeNet/shapenetvox/ShapeNetVox32')
  tmp1, tmp2 = dataset.train.next_batch(2)
  print(dataset.train.volumes.shape)
  print(dataset.validation.volumes.shape)
  print(dataset.test.volumes.shape)
  print(tmp1.shape)
  print(np.amax(tmp1))
  print(np.amin(tmp1))
  print(tmp2)

def main(argv=None):  # pylint: disable=unused-argument
  test()

if __name__ == '__main__':
  main() 
