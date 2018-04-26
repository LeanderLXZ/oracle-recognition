from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import gzip
import shutil
import pickle
import tarfile

import numpy as np
import tensorflow as tf
from os.path import isdir
from tqdm import tqdm
from urllib.request import urlretrieve


def save_data_to_pkl(data, data_path, verbose=True):
  """
  Save data to pickle file.
  """
  file_size = sys.getsizeof(data)
  if file_size / (10**9) > 4:
    if verbose:
      print('File is too large for pickle to save: {:.4}Gb'.format(
          file_size / (10**9)))
    n_parts = int(((file_size / (10**9)) // 4) + 1)
    save_large_data_to_pkl(data, data_path[:-2], n_parts, verbose)
  else:
    with open(data_path, 'wb') as f:
      if verbose:
        print('Saving {}... Size: {:.4}Mb'.format(f.name, file_size / (10**6)))
      pickle.dump(data, f)


def load_data_from_pkl(data_path, verbose=True):
  """
  Load data from pickle file.
  """
  with open(data_path, 'rb') as f:
    if verbose:
      print('Loading {}...'.format(f.name))
    return pickle.load(f)


def save_large_data_to_pkl(data, data_path, n_parts=2, verbose=True):
  """
  Save large data to pickle file.
  """
  if verbose:
    print('Saving large file into {} parts...'.format(n_parts))
  len_part = len(data) // n_parts
  for i in range(n_parts):
    if i == n_parts - 1:
      data_part = data[i * len_part:]
    else:
      data_part = data[i * len_part:(i + 1) * len_part]
    with open(data_path + '_{}.p'.format(i), 'wb') as f:
      if verbose:
        file_size = sys.getsizeof(data_part)
        print('Saving {}... Size: {:.4}Mb'.format(f.name, file_size / (10**6)))
      pickle.dump(data_part, f)


def load_large_data_to_pkl(data_path, n_parts=2, verbose=True):
  """
  Save large data to pickle file.
  """
  data = []
  for i in range(n_parts):
    with open(data_path + '_{}.p'.format(i), 'rb') as f:
      if verbose:
        print('Loading {}...'.format(f.name))
      data.append(pickle.load(f))
  concat = np.concatenate(data, axis=0)
  assert concat.shape[1:] == data[0].shape[1:]
  return concat


def get_vec_length(vec, batch_size, epsilon):
  """
  Get the length of a vector.
  """
  vec_shape = vec.get_shape().as_list()
  num_caps = vec_shape[1]
  vec_dim = vec_shape[2]

  # vec shape: (batch_size, num_caps, vec_dim)
  assert vec.get_shape() == (batch_size, num_caps, vec_dim)

  vec_length = tf.reduce_sum(tf.square(vec), axis=2, keep_dims=True) + epsilon
  vec_length = tf.sqrt(tf.squeeze(vec_length))
  # vec_length shape: (batch_size, num_caps)
  assert vec_length.get_shape() == (batch_size, num_caps)

  return vec_length


def check_dir(path_list):
  """
  Check if directories exit or not.
  """
  for dir_path in path_list:
    if not isdir(dir_path):
      os.makedirs(dir_path)


def thin_line():
  print('-' * 55)


def thick_line():
  print('=' * 55)


def get_batches(x, y, batch_size):
  """
  Split features and labels into batches.
  """
  for start in range(0, len(x) - batch_size, batch_size):
    end = start + batch_size
    yield x[start:end], y[start:end]


def print_status(epoch_i, epochs, step, start_time, loss_train,
                 clf_loss_train, rec_loss_train, acc_train, loss_valid,
                 clf_loss_valid, rec_loss_valid, acc_valid, with_rec):
  """
  Print information while training.
  """
  if with_rec:
    print('Epoch: {}/{} |'.format(epoch_i + 1, epochs),
          'Batch: {} |'.format(step),
          'Time: {:.2f}s |'.format(time.time() - start_time),
          'T_Lo: {:.4f} |'.format(loss_train),
          'T_Cls_Lo: {:.4f} |'.format(clf_loss_train),
          'T_Rec_Lo: {:.4f} |'.format(rec_loss_train),
          'T_Acc: {:.2f}% |'.format(acc_train * 100),
          'V_Lo: {:.4f} |'.format(loss_valid),
          'V_Cls_Lo: {:.4f} |'.format(clf_loss_valid),
          'V_Rec_Lo: {:.4f} |'.format(rec_loss_valid),
          'V_Acc: {:.2f}% |'.format(acc_valid * 100))
  else:
    print('Epoch: {}/{} |'.format(epoch_i + 1, epochs),
          'Batch: {} |'.format(step),
          'Time: {:.2f}s |'.format(time.time() - start_time),
          'Train_Loss: {:.4f} |'.format(loss_train),
          'Train_Acc: {:.2f}% |'.format(acc_train * 100),
          'Valid_Loss: {:.4f} |'.format(loss_valid),
          'Valid_Acc: {:.2f}% |'.format(acc_valid * 100))


def print_full_set_eval(epoch_i, epochs, step, start_time,
                        loss_train, clf_loss_train, rec_loss_train, acc_train,
                        loss_valid, clf_loss_valid, rec_loss_valid, acc_valid,
                        with_full_set_eval, with_rec):
  """
  Print information of full set evaluation.
  """
  thin_line()
  print('Epoch: {}/{} |'.format(epoch_i + 1, epochs),
        'Batch: {} |'.format(step),
        'Time: {:.2f}s |'.format(time.time() - start_time))
  thin_line()
  if with_full_set_eval:
    print('Full_Set_Train_Loss: {:.4f}'.format(loss_train))
    if with_rec:
      print('Train_Classifier_Loss: {:.4f}\n'.format(clf_loss_train),
            'Train_Reconstruction_Loss: {:.4f}'.format(rec_loss_train))
    print('Full_Set_Train_Accuracy: {:.2f}%'.format(acc_train * 100))
  print('Full_Set_Valid_Loss: {:.4f}'.format(loss_valid))
  if with_rec:
    print('Valid_Classifier_Loss: {:.4f}\n'.format(clf_loss_valid),
          'Reconstruction_Valid_Loss: {:.4f}'.format(rec_loss_valid))
  print('Full_Set_Valid_Accuracy: {:.2f}%'.format(acc_valid * 100))


def save_config_log(file_path, cfg, clf_arch_info=None, rec_arch_info=None):
  """
  Save configuration of training.
  """
  file_path = os.path.join(file_path, 'config_log.txt')
  thick_line()
  print('Saving {}...'.format(file_path))

  with open(file_path, 'a') as f:
    local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
    f.write('=' * 55 + '\n')
    f.write('Time: {}\n'.format(local_time))
    f.write('=' * 55 + '\n')
    for key in cfg.keys():
      f.write('{}: {}\n'.format(key, cfg[key]))
    if clf_arch_info is not None:
      f.write('=' * 55 + '\n')
      f.write('Classifier Architecture:\n')
      f.write('-' * 55 + '\n')
      for i, (clf_name, clf_params, clf_shape) in enumerate(clf_arch_info):
        f.write('[{}] {}\n\tParameters: {}\n\tOutput tensor shape: {}\n'.format(
            i, clf_name, clf_params, clf_shape))
    if rec_arch_info is not None:
      f.write('=' * 55 + '\n')
      f.write('Reconstruction Architecture:\n')
      f.write('-' * 55 + '\n')
      for j, (rec_name, rec_params, rec_shape) in enumerate(rec_arch_info):
        f.write('[{}] {}\n\tParameters: {}\n\tOutput tensor shape: {}\n'.format(
            j, rec_name, rec_params, rec_shape))
    f.write('=' * 55)


def save_log(file_path, epoch_i, step, using_time,
             loss_train, clf_loss_train, rec_loss_train, acc_train,
             loss_valid, clf_loss_valid, rec_loss_valid, acc_valid, with_rec):
  """
  Save losses and accuracies while training.
  """
  if with_rec:
    if not os.path.isfile(file_path):
      with open(file_path, 'w') as f:
        header = ['Local_Time', 'Epoch', 'Batch', 'Time', 'Train_Loss',
                  'Train_Classifier_loss', 'Train_Reconstruction_Loss',
                  'Train_Accuracy', 'Valid_Loss', 'Valid_Classifier_loss',
                  'Valid_Reconstruction_Loss', 'Valid_Accuracy']
        writer = csv.writer(f)
        writer.writerow(header)

    with open(file_path, 'a') as f:
      local_time = time.strftime(
          '%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
      log = [local_time, epoch_i, step, using_time,
             loss_train, clf_loss_train, rec_loss_train, acc_train,
             loss_valid, clf_loss_valid, rec_loss_valid, acc_valid]
      writer = csv.writer(f)
      writer.writerow(log)
  else:
    if not os.path.isfile(file_path):
      with open(file_path, 'w') as f:
        header = ['Local_Time', 'Epoch', 'Batch', 'Time', 'Train_Loss',
                  'Train_Accuracy', 'Valid_Loss', 'Valid_Accuracy']
        writer = csv.writer(f)
        writer.writerow(header)

    with open(file_path, 'a') as f:
      local_time = time.strftime(
          '%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
      log = [local_time, epoch_i, step, using_time,
             loss_train, acc_train, loss_valid, acc_valid]
      writer = csv.writer(f)
      writer.writerow(log)


def save_test_log(file_path, loss_test, acc_test,
                  clf_loss_test, rec_loss_test, with_rec):
  """
  Save losses and accuracies of testing.
  """
  file_path = os.path.join(file_path, 'test_log.txt')
  thick_line()
  print('Saving {}...'.format(file_path))

  with open(file_path, 'a') as f:
    local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
    f.write('=' * 55 + '\n')
    f.write('Time: {}\n'.format(local_time))
    f.write('-' * 55 + '\n')
    f.write('Test_Loss: {:.4f}\n'.format(loss_test))
    f.write('Test_Accuracy: {:.2f}%\n'.format(acc_test * 100))
    if with_rec:
      f.write('Test_Train_Loss: {:.4f}\n'.format(clf_loss_test))
      f.write('Test_Reconstruction_Loss: {:.4f}\n'.format(rec_loss_test))
    f.write('=' * 55)


def _read32(bytestream):
  """
  Read 32-bit integer from bytesteam

  Args:
    bytestream: A bytestream
  Returns:
    32-bit integer
  """
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_image(save_path, extract_path):
  """
  Extract the images into a 4D unit8 numpy array [index, y, x, depth].
  """
  # Get data from save_path
  with open(save_path, 'rb') as f:

    print('Extracting {}...'.format(f.name))

    with gzip.GzipFile(fileobj=f) as bytestream:

      magic = _read32(bytestream)
      if magic != 2051:
        raise ValueError(
            'Invalid magic number {} in file: {}'.format(magic, f.name))
      num_images = _read32(bytestream)
      rows = _read32(bytestream)
      cols = _read32(bytestream)
      buf = bytestream.read(rows * cols * num_images)
      data = np.frombuffer(buf, dtype=np.uint8)
      data = data.reshape(num_images, rows, cols, 1)
      save_data_to_pkl(data, extract_path + '.p')


def extract_labels_mnist(save_path, extract_path):
  """
  Extract the labels into a 1D uint8 numpy array [index].
  """
  # Get data from save_path
  with open(save_path, 'rb') as f:

    print('Extracting {}...'.format(f.name))

    with gzip.GzipFile(fileobj=f) as bytestream:

      magic = _read32(bytestream)
      if magic != 2049:
        raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                         (magic, f.name))
      num_items = _read32(bytestream)
      buf = bytestream.read(num_items)
      labels = np.frombuffer(buf, dtype=np.uint8)
      save_data_to_pkl(labels, extract_path + '.p')


def download_and_extract_mnist(url, save_path, extract_path, data_type):

  if not os.path.exists(save_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
      urlretrieve(url, save_path, pbar.hook)

  try:
    if data_type == 'images':
      extract_image(save_path, extract_path)
    elif data_type == 'labels':
      extract_labels_mnist(save_path, extract_path)
    else:
      raise ValueError('Wrong data_type!')
  except Exception as err:
    # Remove extraction folder if there is an error
    shutil.rmtree(extract_path)
    raise err

  # Remove compressed data
  os.remove(save_path)


def load_cifar10_batch(dataset_path, mode, batch_id=None):
  """
  Load a batch of the dataset
  """
  if mode == 'train':
    with open(dataset_path + '/data_batch_' + str(batch_id),
              mode='rb') as file:
      batch = pickle.load(file, encoding='latin1')
  elif mode == 'test':
    with open(dataset_path + '/test_batch',
              mode='rb') as file:
      batch = pickle.load(file, encoding='latin1')
  else:
    raise ValueError('Wrong mode!')

  features = batch['data'].reshape(
      (len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
  labels = batch['labels']

  return features, np.array(labels)


def download_and_extract_cifar10(url, save_path, file_name, extract_path):

  archive_save_path = os.path.join(save_path, file_name)
  extracted_dir_path = os.path.join(save_path, 'cifar-10-batches-py')

  if not os.path.exists(os.path.join(save_path, 'cifar10')):
      with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
        urlretrieve(url, archive_save_path, pbar.hook)
  else:
    raise ValueError('Files already exist!')

  try:
    if not os.path.exists(extracted_dir_path):
      with tarfile.open(archive_save_path) as tar:
        tar.extractall(extract_path)
        tar.close()
  except Exception as err:
    # Remove extraction folder if there is an error
    shutil.rmtree(extract_path)
    raise err

  # Extract images and labels from batches
  features = []
  labels = []
  for batch_i in range(1, 6):
    features_i, labels_i = load_cifar10_batch(
        extracted_dir_path, 'train', batch_i)
    features.append(features_i)
    labels.append(labels_i)
  train_images = np.concatenate(features, axis=0)
  train_labels = np.concatenate(labels, axis=0)
  test_images, test_labels = load_cifar10_batch(
      extracted_dir_path, 'test')

  # Save concatenated images and labels to pickles
  pickle_path = os.path.join(save_path, 'cifar10')
  check_dir([pickle_path])
  save_data_to_pkl(train_images, pickle_path + '/train_images.p')
  save_data_to_pkl(train_labels, pickle_path + '/train_labels.p')
  save_data_to_pkl(test_images, pickle_path + '/test_images.p')
  save_data_to_pkl(test_labels, pickle_path + '/test_labels.p')

  # Remove compressed data
  os.remove(archive_save_path)
  shutil.rmtree(extracted_dir_path)


class DLProgress(tqdm):
  """
  Handle Progress Bar while Downloading
  """
  last_block = 0

  def hook(self, block_num=1, block_size=1, total_size=None):
    """
    A hook function that will be called once on establishment of the network
    connection and once after each block read thereafter.

    Args:
      block_num: A count of blocks transferred so far
      block_size: Block size in bytes
      total_size: The total size of the file. This may be -1 on older FTP
                  servers which do not return a file size in response to a
                  retrieval request.
    """
    self.total = total_size
    self.update((block_num - self.last_block) * block_size)
    self.last_block = block_num
