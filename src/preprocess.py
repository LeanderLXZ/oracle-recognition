from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from os.path import join
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

from models import utils
from config import config as cfg_1
from config_pipeline import config as cfg_2


class DataPreProcess(object):

  def __init__(self, cfg):
    """
    Preprocess data and save as pickle files.

    Args:
      cfg: configuration
    """
    self.cfg = cfg
    self.data_base_name = None
    self.preprocessed_path = None
    self.source_data_path = None

  def _load_data(self):
    """
    Load data set from files.
    """
    utils.thin_line()
    print('Loading {} data set...'.format(self.data_base_name))

    self.x = utils.load_data_from_pkl(
        join(self.source_data_path, 'train_images.p'))
    self.y = utils.load_data_from_pkl(
        join(self.source_data_path, 'train_labels.p'))
    self.x_test = utils.load_data_from_pkl(
        join(self.source_data_path, 'test_images.p'))
    self.y_test = utils.load_data_from_pkl(
        join(self.source_data_path, 'test_labels.p'))

  def _augment_data(self):
    """
    Augment data set and add noises.
    """
    pass

  def _shuffle(self):
    """
    Shuffle data sets.
    """
    utils.thin_line()
    print('Shuffling images and labels...')
    self.x, self.y = shuffle(
        self.x, self.y, random_state=0)
    self.x_test, self.y_test = shuffle(
        self.x_test, self.y_test, random_state=0)

  def _scaling(self):
    """
    Scaling input images to (0, 1).
    """
    utils.thin_line()
    print('Scaling features...')
    
    self.x = np.divide(self.x, 255.)
    self.x_test = np.divide(self.x_test, 255.)

  def _one_hot_encoding(self):
    """
    Scaling images to (0, 1).
    """
    utils.thin_line()
    print('One-hot-encoding labels...')
    
    encoder = LabelBinarizer()
    encoder.fit(self.y)
    self.y = encoder.transform(self.y)
    self.y_test = encoder.transform(self.y_test)

  def _split_data(self):
    """
    Split data set for training, validation and testing.
    """
    utils.thin_line()
    print('Splitting train/valid/test set...')
    
    if self.data_base_name == 'mnist':
      train_stop = 55000
    elif self.data_base_name == 'cifar10':
      train_stop = 45000
    else:
      raise ValueError('Wrong database name!')

    if self.cfg.DPP_TEST_AS_VALID:
      self.x_train = self.x
      self.y_train = self.y
      self.x_valid = self.x_test
      self.y_valid = self.y_test
    else:
      self.x_train = self.x[:train_stop]
      self.y_train = self.y[:train_stop]
      self.x_valid = self.x[train_stop:]
      self.y_valid = self.y[train_stop:]

  def _check_data(self):
    """
    Check data format.
    """
    assert self.x_train.max() <= 1, self.x_train.max()
    assert self.y_train.max() <= 1, self.y_train.max()
    assert self.x_valid.max() <= 1, self.x_valid.max()
    assert self.y_valid.max() <= 1, self.y_valid.max()
    assert self.x_test.max() <= 1, self.x_test.max()
    assert self.y_test.max() <= 1, self.y_test.max()

    assert self.x_train.min() >= 0, self.x_train.min()
    assert self.y_train.min() >= 0, self.y_train.min()
    assert self.x_valid.min() >= 0, self.x_valid.min()
    assert self.y_valid.min() >= 0, self.y_valid.min()
    assert self.x_test.min() >= 0, self.x_test.min()
    assert self.y_test.min() >= 0, self.y_test.min()

    if self.cfg.DPP_TEST_AS_VALID:
      if self.data_base_name == 'mnist':
        assert self.x_train.shape == (60000, 28, 28, 1), self.x_train.shape
        assert self.y_train.shape == (60000, 10), self.y_train.shape
        assert self.x_valid.shape == (10000, 28, 28, 1), self.x_valid.shape
        assert self.y_valid.shape == (10000, 10), self.y_valid.shape
        assert self.x_test.shape == (10000, 28, 28, 1), self.x_test.shape
        assert self.y_test.shape == (10000, 10), self.y_test.shape
      elif self.data_base_name == 'cifar10':
        assert self.x_train.shape == (50000, 32, 32, 3), self.x_train.shape
        assert self.y_train.shape == (50000, 10), self.y_train.shape
        assert self.x_valid.shape == (10000, 32, 32, 3), self.x_valid.shape
        assert self.y_valid.shape == (10000, 10), self.y_valid.shape
        assert self.x_test.shape == (10000, 32, 32, 3), self.x_test.shape
        assert self.y_test.shape == (10000, 10), self.y_test.shape
      else:
        raise ValueError('Wrong database name!')

    else:
      if self.data_base_name == 'mnist':
        assert self.x_train.shape == (55000, 28, 28, 1), self.x_train.shape
        assert self.y_train.shape == (55000, 10), self.y_train.shape
        assert self.x_valid.shape == (5000, 28, 28, 1), self.x_valid.shape
        assert self.y_valid.shape == (5000, 10), self.y_valid.shape
        assert self.x_test.shape == (10000, 28, 28, 1), self.x_test.shape
        assert self.y_test.shape == (10000, 10), self.y_test.shape
      elif self.data_base_name == 'cifar10':
        assert self.x_train.shape == (45000, 32, 32, 3), self.x_train.shape
        assert self.y_train.shape == (45000, 10), self.y_train.shape
        assert self.x_valid.shape == (5000, 32, 32, 3), self.x_valid.shape
        assert self.y_valid.shape == (5000, 10), self.y_valid.shape
        assert self.x_test.shape == (10000, 32, 32, 3), self.x_test.shape
        assert self.y_test.shape == (10000, 10), self.y_test.shape
      else:
        raise ValueError('Wrong database name!')

  def _save_data(self):
    """
    Save data set to pickle files.
    """
    utils.thin_line()
    print('Saving pickle files...')

    utils.check_dir([self.preprocessed_path])
    
    utils.save_data_to_pkl(
        self.x_train, join(self.preprocessed_path, 'x_train.p'))
    utils.save_data_to_pkl(
        self.y_train, join(self.preprocessed_path, 'y_train.p'))
    utils.save_data_to_pkl(
        self.x_valid, join(self.preprocessed_path, 'x_valid.p'))
    utils.save_data_to_pkl(
        self.y_valid, join(self.preprocessed_path, 'y_valid.p'))
    utils.save_data_to_pkl(
        self.x_test, join(self.preprocessed_path, 'x_test.p'))
    utils.save_data_to_pkl(
        self.y_test, join(self.preprocessed_path, 'y_test.p'))

  def pipeline(self, data_base_name):
    """
    Pipeline of preprocessing data.

    Arg:
      data_base_name: name of data base
    """
    utils.thick_line()
    print('Start Preprocessing...')

    start_time = time.time()

    self.data_base_name = data_base_name
    self.preprocessed_path = join(self.cfg.DPP_DATA_PATH, data_base_name)
    self.source_data_path = join(self.cfg.SOURCE_DATA_PATH, data_base_name)

    # Load data
    self._load_data()

    # Augment data
    self._augment_data()

    # Shuffle data set
    # self._shuffle()

    # Scaling images to (0, 1)
    self._scaling()

    # One-hot-encoding labels
    self._one_hot_encoding()

    # Split data set into train/valid/test
    self._split_data()

    # Check data format.
    self._check_data()

    # Save data to pickles
    self._save_data()

    utils.thin_line()
    print('Done! Using {:.3}s'.format(time.time() - start_time))
    utils.thick_line()


if __name__ == '__main__':

  utils.thick_line()
  print('Input [ 1 ] to preprocess the MNIST database.')
  print('Input [ 2 ] to preprocess the CIFAR-10 database.')
  print("Input [ 3 ] to preprocess the MNIST and CIFAR-10 database.")
  utils.thin_line()
  input_mode = input('Input: ')

  utils.thick_line()
  print('Input [ 1 ] to use config.')
  print('Input [ 2 ] to use config_pipeline.')
  utils.thin_line()
  input_cfg = input('Input: ')

  if input_cfg == '1':
    DPP = DataPreProcess(cfg_1)
  elif input_cfg == '2':
    DPP = DataPreProcess(cfg_2)
  else:
    raise ValueError('Wrong config input! Found: {}'.format(input_cfg))

  if input_mode == '1':
    DPP.pipeline('mnist')
  elif input_mode == '2':
    DPP.pipeline('cifar10')
  elif input_mode == '3':
    DPP.pipeline('mnist')
    DPP.pipeline('cifar10')
  else:
    raise ValueError('Wrong database input! Found: {}'.format(input_mode))
