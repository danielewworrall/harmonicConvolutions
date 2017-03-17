"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import os
import numpy as np
from npyio import NpyTarReader

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class_names_40 = ['airplane', 'bathtub', 'bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup', 'curtain', 'desk', 'door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person', 'piano', 'plant', 'radio','range_hood','sink','sofa', 'stairs', 'stool', 'table','tent','toilet','tv_stand', 'vase', 'wardrobe', 'xbox']

class_id_to_name = {
       1: "bathtub",
       2: "bed",
       3: "chair",
       4: "desk",
       5: "dresser",
       6: "monitor",
       7: "night_stand",
       8: "sofa",
       9: "table",
       10: "toilet"
}

class DataSet(object):
  def __init__(self, path, one_hot, skipchance=0, sel=None):
    reader = NpyTarReader(path)
    num_examples = reader.num_files
    valid_counter = 0

    if sel is None:
        dosel = False
        sel = []
    else:
        dosel = True

    for ix, (x, name) in enumerate(reader):
        if ix==0:
            # read first file to get width, height, depth
            depth, height, width = x.shape # TODO not sure if order is correct
            volumes = np.empty([num_examples, depth, height, width, 1], dtype=np.uint8)
            labels = -1000*np.ones([num_examples], dtype=np.int16)
        cur_id = int(name[0:3])
        #tmpname = class_id_to_name.get(cur_id)
        #val1 = name[4:7]
        #val2 = tmpname[:3]
        #if val1!=val2:
        #    print(val1, val2)

        label = cur_id-1
        if label is not None:
            if dosel:
                if ix in sel:
                    volumes[valid_counter,:,:,:,0] = x
                    labels[valid_counter] = label
                    valid_counter += 1
            else:
                if np.random.rand()<skipchance:
                    sel.append(ix)
                else:
                    volumes[valid_counter,:,:,:,0] = x
                    labels[valid_counter] = label
                    valid_counter += 1
        else:
            print('Unexpected class name:', name)

    self.sel = sel
    self._volumes = volumes[0:valid_counter]
    self._labels = labels[0:valid_counter]
    self._num_examples = valid_counter
    self.one_hot = one_hot

    max_lab = np.amax(self._labels)+1
    print(max_lab)
    balance = np.zeros(max_lab)
    for i in range(max_lab):
        balance[i] = np.sum((self._labels==i).astype(np.float32))/self._labels.shape[0]

    balance = np.array(balance, dtype=np.float32)
    self.class_balance = balance[np.newaxis,:]

    print('num_examples: ', self._num_examples)
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
      labels = labels.astype(np.int32)
      if self.one_hot:
          #print(labels)
          labels = dense_to_one_hot(labels)
          #print(labels)
      return volumes, labels

def read_data_sets(basedir, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  basedir = os.path.realpath(os.path.expanduser(basedir))
  print('Reading', basedir)

  data_sets.train = DataSet(os.path.join(basedir, 'shapenet10_train_nr.tar'), one_hot, 0.05)
  data_sets.validation = DataSet(os.path.join(basedir, 'shapenet10_train_nr.tar'), one_hot, 0.0, data_sets.train.sel)
  #data_sets.validation = data_sets.test # no decision based on validation sets

  data_sets.test = DataSet(os.path.join(basedir, 'shapenet10_test_nr.tar'), one_hot)

  return data_sets
  
def test():
  print(class_id)
  dataset = read_data_sets('~/Documents/Datasets/ModelNet/')
  #dataset = read_data_sets('~/ShapeNet/shapenetvox/ShapeNetVox32')
  #dataset = read_data_sets('~/scratch/Datasets/ModelNet/', True)
  tmp1, tmp2 = dataset.train.next_batch(2)
  print(dataset.train.volumes.shape)
  #print(dataset.validation.volumes.shape)
  print(dataset.test.volumes.shape)
  print(tmp1.shape)
  print(np.amax(tmp1))
  print(np.amin(tmp1))
  print(tmp2)
  print(np.unique(dataset.train._labels))
  print(dataset.train.class_balance)
  print(dataset.test.class_balance)
  print(np.sum(dataset.train.class_balance))
  print(dataset.train.class_balance.shape)
  print(dataset.test.class_balance.shape)
  #print(np.bincount(dataset.train._labels))
  #print(np.bincount(dataset.validation._labels))
  #print(np.bincount(dataset.test._labels))
  #for i in (np.unique(dataset.train._labels)):
  #    print(class_names[i])
  

def main(argv=None):  # pylint: disable=unused-argument
  test()

if __name__ == '__main__':
  main() 
