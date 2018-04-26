from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
from PIL import Image
import numpy as np
import sklearn.utils
from copy import copy
from tqdm import tqdm
from os.path import join
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from models import utils
from config import config as cfg_1
from config_pipeline import config as cfg_2

from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


class DataPreProcess(object):

  def __init__(self, cfg, seed=None):
    """
    Preprocess data and save as pickle files.

    Args:
      cfg: configuration
    """
    self.cfg = cfg
    self.seed = seed
    self.data_base_name = None
    self.preprocessed_path = None
    self.source_data_path = None
    self.img_size = self.cfg.ORACLE_IMAGE_SIZE

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

  def _load_oracle_radicals(self, data_aug_param=None):
    """
    Load oracle data set from files.
    """
    utils.thin_line()
    print('Loading {} data set...'.format(self.data_base_name))
    classes = sorted(os.listdir(self.source_data_path))

    self.x = []
    self.y = []
    for cls_name in tqdm(
          classes[:self.cfg.NUM_RADICALS], ncols=100, unit='class'):

      # Load images from raw data pictures
      class_dir = join(self.source_data_path, cls_name)
      images = os.listdir(class_dir)
      x_tensor = []
      for img_name in images:
        # Load image
        img = Image.open(join(class_dir, img_name)).convert('L')
        # Reshape image
        reshaped_img = self._reshape_img(img)
        # Save image to array
        x_tensor.append(reshaped_img)

      # Data augment
      if self.cfg.USE_DATA_AUG:
        x_tensor = self._augment_data(x_tensor, data_aug_param)
      assert len(x_tensor) == self.cfg.MAX_IMAGE_NUM

      self.x.append(x_tensor)
      self.y.extend([int(cls_name) for _ in range(len(x_tensor))])

    self.x = np.array(
        self.x, dtype=np.float32).reshape((-1, *self.x[0][0].shape))
    self.y = np.array(self.y, dtype=np.int)

    print('Images shape: {}\nLabels shape: {}'.format(
        self.x.shape, self.y.shape))
    assert len(self.x) == len(self.y)

  def _reshape_img(self, img):
    """
    Reshaping an image to a ORACLE_IMAGE_SIZE
    """
    reshaped_image = Image.new('L', self.img_size, 'white')
    img_width, img_height = img.size

    if img_width > img_height:
      w_s = self.img_size[0]
      h_s = int(w_s * img_height // img_width)
      img = img.resize((w_s, h_s), Image.ANTIALIAS)
      reshaped_image.paste(img, (0, int((self.img_size[1] - h_s) // 2)))
    else:
      h_s = self.img_size[1]
      w_s = int(h_s * img_width // img_height)
      img = img.resize((w_s, h_s), Image.ANTIALIAS)
      reshaped_image.paste(img, (int((self.img_size[0] - w_s) // 2), 0))

    reshaped_image = np.array(reshaped_image, dtype=np.float32)
    reshaped_image = reshaped_image.reshape((*reshaped_image.shape, 1))
    assert reshaped_image.shape == (*self.img_size, 1)
    return reshaped_image

  def _augment_data(self, tensor, data_aug_param):
    """
    Augment data set and add noises.
    """
    data_generator = ImageDataGenerator(**data_aug_param)
    new_x_tensors = copy(tensor)
    while True:
      for i in range(len(tensor)):
        if len(new_x_tensors) >= self.cfg.MAX_IMAGE_NUM:
          return new_x_tensors
        augmented = data_generator.random_transform(tensor[i])
        new_x_tensors.append(augmented)

  def _train_test_split(self):
    """
    Split data set for training and testing.
    """
    utils.thin_line()
    print('Splitting train/test set...')
    self.x, self.x_test, self.y, self.y_test = train_test_split(
        self.x,
        self.y,
        test_size=self.cfg.TEST_SIZE,
        shuffle=True,
        random_state=self.seed
    )

  def _shuffle(self):
    """
    Shuffle data sets.
    """
    utils.thin_line()
    print('Shuffling images and labels...')
    self.x, self.y = sklearn.utils.shuffle(
        self.x, self.y, random_state=self.seed)
    self.x_test, self.y_test = sklearn.utils.shuffle(
        self.x_test, self.y_test, random_state=self.seed)

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

  def _train_valid_split(self):
    """
    Split data set for training and validation
    """
    utils.thin_line()
    print('Splitting train/valid set...')

    train_stop = int(len(self.x) * self.cfg.VALID_SIZE)
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

    def _check_oracle_data():
      n_classes = \
        148 if self.cfg.NUM_RADICALS is None else self.cfg.NUM_RADICALS
      assert self.x_train.shape == (
        len(self.x_train), *self.img_size, 1), self.x_train.shape
      assert self.y_train.shape == (
        len(self.y_train), n_classes), self.y_train.shape
      assert self.x_valid.shape == (
        len(self.x_valid), *self.img_size, 1), self.x_valid.shape
      assert self.y_valid.shape == (
        len(self.y_valid), n_classes), self.y_valid.shape
      assert self.x_test.shape == (
        len(self.x_test), *self.img_size, 1), self.x_test.shape
      assert self.y_test.shape == (
        len(self.y_test), n_classes), self.y_test.shape

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
      elif self.data_base_name == 'radical':
        _check_oracle_data()
      else:
        raise ValueError('Wrong database name!')

    else:
      if self.data_base_name == 'mnist':
        assert self.x_train.shape == (54000, 28, 28, 1), self.x_train.shape
        assert self.y_train.shape == (54000, 10), self.y_train.shape
        assert self.x_valid.shape == (6000, 28, 28, 1), self.x_valid.shape
        assert self.y_valid.shape == (6000, 10), self.y_valid.shape
        assert self.x_test.shape == (10000, 28, 28, 1), self.x_test.shape
        assert self.y_test.shape == (10000, 10), self.y_test.shape
      elif self.data_base_name == 'cifar10':
        assert self.x_train.shape == (45000, 32, 32, 3), self.x_train.shape
        assert self.y_train.shape == (45000, 10), self.y_train.shape
        assert self.x_valid.shape == (5000, 32, 32, 3), self.x_valid.shape
        assert self.y_valid.shape == (5000, 10), self.y_valid.shape
        assert self.x_test.shape == (10000, 32, 32, 3), self.x_test.shape
        assert self.y_test.shape == (10000, 10), self.y_test.shape
      elif self.data_base_name == 'radical':
        _check_oracle_data()
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
    if self.data_base_name == 'mnist' or self.data_base_name == 'cifar10':
      self._load_data()
    elif self.data_base_name == 'radical':
      data_aug_parameters = dict(
          rotation_range=40,
          width_shift_range=0.1,
          height_shift_range=0.1,
          shear_range=0.1,
          zoom_range=0.1,
          horizontal_flip=True,
          fill_mode='nearest'
      )
      self._load_oracle_radicals(data_aug_parameters)
      self._train_test_split()

    # Shuffle data set
    self._shuffle()

    # Scaling images to (0, 1)
    self._scaling()

    # One-hot-encoding labels
    self._one_hot_encoding()

    # Split data set into train/valid
    self._train_valid_split()

    # Check data format.
    self._check_data()

    # Save data to pickles
    self._save_data()

    utils.thin_line()
    print('Done! Using {:.4}s'.format(time.time() - start_time))
    utils.thick_line()


if __name__ == '__main__':

  global_seed = None

  utils.thick_line()
  print('Input [ 1 ] to preprocess the Oracle Radicals database.')
  print('Input [ 2 ] to preprocess the MNIST database.')
  print('Input [ 3 ] to preprocess the CIFAR-10 database.')
  print("Input [ 4 ] to preprocess the MNIST and CIFAR-10 database.")
  utils.thin_line()
  input_mode = input('Input: ')

  utils.thick_line()
  print('Input [ 1 ] to use config.')
  print('Input [ 2 ] to use config_pipeline.')
  utils.thin_line()
  input_cfg = input('Input: ')

  if input_cfg == '1':
    DPP = DataPreProcess(cfg_1, global_seed)
  elif input_cfg == '2':
    DPP = DataPreProcess(cfg_2, global_seed)
  else:
    raise ValueError('Wrong config input! Found: {}'.format(input_cfg))

  if input_mode == '1':
    DPP.pipeline('radical')
  elif input_mode == '2':
    DPP.pipeline('mnist')
  elif input_mode == '3':
    DPP.pipeline('cifar10')
  elif input_mode == '4':
    DPP.pipeline('mnist')
    DPP.pipeline('cifar10')
  else:
    raise ValueError('Wrong database input! Found: {}'.format(input_mode))
