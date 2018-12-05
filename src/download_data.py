from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join

from config import config as cfg
from models import utils


# Download database
def download_data(data_base_name):
  """Download database."""
  utils.thick_line()
  print('Downloading {} data set...'.format(data_base_name))
  utils.thin_line()

  if data_base_name == 'mnist':
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    source_data_path_ = join(cfg.SOURCE_DATA_PATH, 'mnist')
    utils.check_dir([source_data_path_])

    utils.download_and_extract_mnist(
        url=SOURCE_URL + TRAIN_IMAGES,
        save_path=join(source_data_path_, TRAIN_IMAGES),
        extract_path=join(source_data_path_, 'train_images'),
        data_type='images'
    )
    utils.download_and_extract_mnist(
        url=SOURCE_URL + TRAIN_LABELS,
        save_path=join(source_data_path_, TRAIN_LABELS),
        extract_path=join(source_data_path_, 'train_labels'),
        data_type='labels'
    )
    utils.download_and_extract_mnist(
        url=SOURCE_URL + TEST_IMAGES,
        save_path=join(source_data_path_, TEST_IMAGES),
        extract_path=join(source_data_path_, 'test_images'),
        data_type='images'
    )
    utils.download_and_extract_mnist(
        url=SOURCE_URL + TEST_LABELS,
        save_path=join(source_data_path_, TEST_LABELS),
        extract_path=join(source_data_path_, 'test_labels'),
        data_type='labels'
    )

  elif data_base_name == 'cifar10':
    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
    FILE_NAME = 'cifar-10-python.tar.gz'

    utils.check_dir([cfg.SOURCE_DATA_PATH])

    utils.download_and_extract_cifar10(
        url=SOURCE_URL + FILE_NAME,
        save_path=cfg.SOURCE_DATA_PATH,
        file_name=FILE_NAME,
        extract_path=cfg.SOURCE_DATA_PATH
    )

  else:
    raise ValueError('Wrong database name!')

  utils.thick_line()


if __name__ == '__main__':

  utils.thick_line()
  print('Input [ 1 ] to download the MNIST database.')
  print('Input [ 2 ] to download the CIFAR-10 database.')
  print("Input [ 3 ] to download the MNIST and CIFAR-10 database.")
  utils.thin_line()
  input_ = input('Input: ')

  if input_ == '1':
    download_data('mnist')
  elif input_ == '2':
    download_data('cifar10')
  elif input_ == '3':
    download_data('mnist')
    download_data('cifar10')
  else:
    raise ValueError('Wrong input! Found: {}'.format(input_))
