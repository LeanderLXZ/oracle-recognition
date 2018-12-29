from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import math
import argparse
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
from baseline_config import config as basel_cfg

from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


class DataPreProcess(object):

  def __init__(self, cfg, seed=None, data_base_name=None):
    """
    Preprocess data and save as pickle files.

    Args:
      cfg: configuration
    """
    self.cfg = cfg
    self.seed = seed
    self.data_base_name = data_base_name
    self.preprocessed_path = None
    self.source_data_path = None

    if self.data_base_name == 'mnist':
      self.img_size = (28, 28)
      self.img_mode = 'L'
    elif self.data_base_name == 'cifar10':
      self.img_size = (32, 32)
      self.img_mode = 'RGB'
    elif self.data_base_name == 'radical':
      self.img_size = self.cfg.IMAGE_SIZE
      self.img_mode = 'L'
    else:
      raise ValueError('Wrong database name!')

  def _load_data(self, show_img=False):
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

    # Data augment
    if self.cfg.USE_DATA_AUG:
      utils.thin_line()
      print('Augmenting data...'.format(self.data_base_name))

      x_y_dict = self._get_x_y_dict(self.x, self.y)
      x_new = []
      y_new = []
      for y_ in tqdm(x_y_dict.keys(),
                     ncols=100,
                     unit=' class'):
        x_ = x_y_dict[y_]
        x_ = self._augment_data(
            x_,
            self.cfg.DATA_AUG_PARAM,
            img_num=self.cfg.MAX_IMAGE_NUM,
            add_self=self.cfg.DATA_AUG_KEEP_SOURCE)
        x_new.append(x_)
        y_new.extend([int(y_) for _ in range(len(x_))])

      self.x = np.array(
          x_new, dtype=np.float32).reshape((-1, *self.x[0].shape))
      self.y = np.array(y_new, dtype=np.int)

    if show_img:
      self._grid_show_imgs(self.x, self.y, 25, mode='L')

  def _load_oracle_radicals(self, show_img=False):
    """
    Load oracle data set from files.
    """
    utils.thin_line()
    print('Loading {} data set...'.format(self.data_base_name))
    classes = sorted(os.listdir(self.source_data_path))
    if '.DS_Store' in classes:
      classes.remove('.DS_Store')

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
        # Resize image
        reshaped_img = self._resize_oracle_img(img)
        # Change background
        reshaped_img = 255 - reshaped_img
        # Save image to array
        x_tensor.append(reshaped_img)

      # Data augment
      if self.cfg.USE_DATA_AUG:
        x_tensor = self._augment_data(
            x_tensor, self.cfg.DATA_AUG_PARAM, img_num=self.cfg.MAX_IMAGE_NUM)
      assert len(x_tensor) == self.cfg.MAX_IMAGE_NUM

      self.x.append(x_tensor)
      self.y.extend([int(cls_name) for _ in range(len(x_tensor))])

    self.x = np.array(
        self.x, dtype=np.float32).reshape((-1, *self.x[0][0].shape))
    self.y = np.array(self.y, dtype=np.int)

    print('Images shape: {}\nLabels shape: {}'.format(
        self.x.shape, self.y.shape))
    assert len(self.x) == len(self.y)

    if show_img:
      self._grid_show_imgs(self.x, self.y, 25, mode='L')

  @staticmethod
  def _get_x_y_dict(x, y, y_encoded=False):
    """Get y:x dictionary."""
    if y_encoded:
      # [[1, 0, ..., 0], ..., [0, 1, ..., 0]] -> [1, ..., 2]
      y = [np.argmax(y_) for y_ in y]
    classes = set(y)
    x_y_dict = {c: [] for c in classes}
    for idx, y_ in enumerate(y):
      x_y_dict[y_].append(x[idx])
    return x_y_dict

  def _resize_imgs(self):
    """Resize images"""
    self.img_size = self.cfg.IMAGE_SIZE
    self.x = utils.img_resize(
        self.x, self.cfg.IMAGE_SIZE, img_mode=self.img_mode,
        resize_filter=Image.ANTIALIAS)
    self.x_test = utils.img_resize(
        self.x_test, self.cfg.IMAGE_SIZE, img_mode=self.img_mode,
        resize_filter=Image.ANTIALIAS)

  def _resize_oracle_img(self, img):
    """
    Resizing an image to IMAGE_SIZE
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

  @staticmethod
  def _augment_data(tensor, data_aug_param, img_num, add_self=True):
    """
    Augment data set and add noises.
    """
    data_generator = ImageDataGenerator(**data_aug_param)
    if add_self:
      new_x_tensors = copy(tensor)
    else:
      new_x_tensors = []
    while True:
      for i in range(len(tensor)):
        if len(new_x_tensors) >= img_num:
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
    One-hot-encoding labels.
    """
    utils.thin_line()
    print('One-hot-encoding labels...')
    
    encoder = LabelBinarizer()
    encoder.fit(self.y)
    self.y = encoder.transform(self.y)
    self.y_test = encoder.transform(self.y_test)

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

  def _generate_multi_obj_img(self,
                              show_img=False,
                              data_aug=False):
    """
    Generate images of superpositions of multi-objects
    """
    utils.thin_line()
    print('Generating images of superpositions of multi-objects...')
    self.x_test_mul = []
    self.y_test_mul = []

    for _ in tqdm(range(self.cfg.NUM_MULTI_IMG), ncols=100, unit=' images'):
      # Get images for merging
      if self.cfg.REPEAT:
        # Repetitive labels
        mul_img_idx_ = np.random.choice(
            len(self.x_test), self.cfg.NUM_MULTI_OBJECT, replace=False)
        mul_imgs = list(self.x_test[mul_img_idx_])
        mul_y = [0 if y_ == 0 else 1 for y_ in np.sum(
            self.y_test[mul_img_idx_], axis=0)]
      else:
        # No repetitive labels
        x_y_dict = self._get_x_y_dict(self.x_test, self.y_test, y_encoded=True)
        y_list = np.random.choice(
            list(x_y_dict.keys()), self.cfg.NUM_MULTI_OBJECT, replace=False)
        mul_imgs = []
        mul_y = []
        for y_ in y_list:
          x_ = x_y_dict[y_]
          x_ = x_[np.random.choice(len(x_))]
          mul_imgs.append(x_)
          mul_y.append(y_)
        mul_y = [1 if i in mul_y else 0 for i in range(len(x_y_dict.keys()))]

      # Data augment
      if data_aug:
        mul_imgs = np.array(self._augment_data(
            mul_imgs,
            self.cfg.DATA_AUG_PARAM,
            img_num=len(mul_imgs),
            add_self=False))

      # Merge images
      if self.cfg.OVERLAP:
        mul_imgs = utils.img_add_overlap(mul_imgs, merge=False, gamma=0)
      else:
        mul_imgs = utils.img_add_no_overlap(
            mul_imgs, self.cfg.NUM_MULTI_OBJECT,
            img_mode=self.img_mode, resize_filter=Image.ANTIALIAS)

      self.x_test_mul.append(mul_imgs)
      self.y_test_mul.append(mul_y)

    self.x_test_mul = np.array(self.x_test_mul)
    self.y_test_mul = np.array(self.y_test_mul)

    if show_img:
      y_show = np.argsort(
          self.y_test_mul, axis=1)[:, -self.cfg.NUM_MULTI_OBJECT:]
      self._grid_show_imgs(self.x_test_mul, y_show, 25, mode='L')

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

    if self.data_base_name == 'mnist':
      train_num = 60000
      test_num = 10000
      n_classes = 10
      img_size = (*self.img_size, 1)

    elif self.data_base_name == 'cifar10':
      train_num = 50000
      test_num = 10000
      n_classes = 10
      img_size = (*self.img_size, 3)

    elif self.data_base_name == 'radical':
      train_num = len(self.x_train)
      test_num = len(self.y_train)
      n_classes = \
          148 if self.cfg.NUM_RADICALS is None else self.cfg.NUM_RADICALS
      img_size = (*self.cfg.IMAGE_SIZE, 1)
    else:
      raise ValueError('Wrong database name!')

    if self.cfg.USE_DATA_AUG:
      train_num = n_classes * self.cfg.MAX_IMAGE_NUM

    if self.cfg.DPP_TEST_AS_VALID:
      valid_num = test_num
    else:
      train_num = (train_num - test_num) * (1 - self.cfg.VALID_SIZE)
      valid_num = (train_num - test_num) * self.cfg.VALID_SIZE

    assert self.x_train.shape == (train_num, *img_size), self.x_train.shape
    assert self.y_train.shape == (train_num, n_classes), self.y_train.shape
    assert self.x_valid.shape == (valid_num, *img_size), self.x_valid.shape
    assert self.y_valid.shape == (valid_num, n_classes), self.y_valid.shape
    assert self.x_test.shape == (test_num, *img_size), self.x_test.shape
    assert self.y_test.shape == (test_num, n_classes), self.y_test.shape

    if self.cfg.NUM_MULTI_OBJECT:
      assert self.x_test_mul.max() <= 1, self.x_test_mul.max()
      assert self.y_test_mul.max() <= 1, self.y_test_mul
      assert self.x_test_mul.min() >= 0, self.x_test_mul.min()
      assert self.y_test_mul.min() >= 0, self.y_test_mul
      assert self.x_test_mul.shape == (
        self.cfg.NUM_MULTI_IMG, *img_size), self.x_test_mul.shape
      assert self.y_test_mul.shape == (
        self.cfg.NUM_MULTI_IMG, n_classes), self.y_test_mul.shape

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

    if self.cfg.NUM_MULTI_OBJECT:
      utils.save_data_to_pkl(
          self.x_test_mul, join(self.preprocessed_path, 'x_test_mul.p'))
      utils.save_data_to_pkl(
          self.y_test_mul, join(self.preprocessed_path, 'y_test_mul.p'))

  @staticmethod
  def _grid_show_imgs(x, y, n_img_show, mode='L'):
    sample_idx_ = np.random.choice(
        len(y), n_img_show, replace=False)
    utils.square_grid_show_imgs(x[sample_idx_], mode=mode)
    y_show = y[sample_idx_]
    size = math.floor(np.sqrt(n_img_show))
    print(y_show.reshape(size, size, -1))

  def pipeline(self):
    """
    Pipeline of preprocessing data.

    Arg:
      data_base_name: name of data base
    """
    utils.thick_line()
    print('Start Preprocessing...')

    start_time = time.time()

    self.preprocessed_path = join(self.cfg.DPP_DATA_PATH, self.data_base_name)
    self.source_data_path = join(self.cfg.SOURCE_DATA_PATH, self.data_base_name)

    # show_img = True
    show_img = False

    # Load data
    if self.data_base_name == 'mnist' or self.data_base_name == 'cifar10':
      self._load_data(show_img=show_img)
      if self.cfg.RESIZE_IMG:
        self._resize_imgs()
    elif self.data_base_name == 'radical':
      self._load_oracle_radicals(show_img=show_img)
      self._train_test_split()

    # Scaling images to (0, 1)
    self._scaling()

    # One-hot-encoding labels
    self._one_hot_encoding()

    # Shuffle data set
    self._shuffle()

    # Generate multi-objects test images
    if self.cfg.NUM_MULTI_OBJECT:
      self._generate_multi_obj_img(show_img=show_img, data_aug=False)

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

  parser = argparse.ArgumentParser(
      description="Testing the model."
  )
  parser.add_argument('-b', '--baseline', action="store_true",
                      help="Use baseline configurations.")
  args = parser.parse_args()

  if args.baseline:
    utils.thick_line()
    print('Running baseline model.')
    DataPreProcess(basel_cfg, global_seed, 'mnist').pipeline()
  else:
    utils.thick_line()
    print('Input [ 1 ] to preprocess the Oracle Radicals database.')
    print('Input [ 2 ] to preprocess the MNIST database.')
    print('Input [ 3 ] to preprocess the CIFAR-10 database.')
    utils.thin_line()
    input_mode = input('Input: ')
    utils.thick_line()
    print('Input [ 1 ] to use config.')
    print('Input [ 2 ] to use config_pipeline.')
    utils.thin_line()
    input_cfg = input('Input: ')

    if input_cfg == '1':
      cfg_selected = cfg_1
    elif input_cfg == '2':
      cfg_selected = cfg_2
    else:
      raise ValueError('Wrong config input! Found: {}'.format(input_cfg))

    if input_mode == '1':
      DataPreProcess(cfg_selected, global_seed, 'radical').pipeline()
    elif input_mode == '2':
      DataPreProcess(cfg_selected, global_seed, 'mnist').pipeline()
    elif input_mode == '3':
      DataPreProcess(cfg_selected, global_seed, 'cifar10').pipeline()
    else:
      raise ValueError('Wrong database input! Found: {}'.format(input_mode))
