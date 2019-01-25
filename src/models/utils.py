from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
import re
import csv
import math
import time
import gzip
import shutil
import pickle
import tarfile
from os import listdir

import numpy as np
import tensorflow as tf
from os.path import join, isdir
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from urllib.request import urlretrieve


def save_data_to_pkl(data, data_path, verbose=True):
  """data to pickle file."""
  file_size = data.nbytes
  if file_size / (10**9) > 4:
    if verbose:
      print('Saving {}...'.format(data_path))
      print('File is too large (>4Gb) for pickle to save: {:.4}Gb'.format(
          file_size / (10**9)))
    save_large_data_to_pkl(data, data_path[:-2], verbose)
  else:
    with open(data_path, 'wb') as f:
      if verbose:
        print('Saving {}...'.format(f.name))
        print('Shape: {}'.format(np.array(data).shape))
        print('Size: {:.4}Mb'.format(file_size / (10**6)))
      pickle.dump(data, f)


def save_large_data_to_pkl(data, data_path, verbose=True):
  """Save large data to pickle file."""
  file_size = data.nbytes
  max_part_size = 2**31 - 1  # 2Gb
  n_parts = file_size // max_part_size + 1
  len_part = int(((len(data) // n_parts) // 2048) * 2048)
  n_parts = int(len(data) // len_part + 1)

  if verbose:
    print('Saving {} into {} parts...'.format(data_path, n_parts))
    print('Total Size: {:.4}Gb'.format(file_size / (10**9)))
    print('Data Shape: ', data.shape)

  for i in range(n_parts):
    if i == n_parts - 1:
      data_part = data[i * len_part:]
    else:
      data_part = data[i * len_part:(i + 1) * len_part]
    data_path_i = data_path + '_{}.p'.format(i)
    with open(data_path_i, 'wb') as f:
      if verbose:
        file_size_part = data_part.nbytes
        print('Saving {}... Part Size: {:.4}Gb'.format(
            f.name, file_size_part / (10**9)))
        print('Part Shape: ', data_part.shape)
      pickle.dump(data_part, f)
    del data_part
    gc.collect()


def load_pkls(dir_path, file_name, verbose=True):
  """Load data from pickle file or files."""
  indices = []
  for f_name in listdir(dir_path):
    m = re.match(file_name + '_(\d*).p', f_name)
    if m:
      indices.append(int(m.group(1)))
  if indices:
    return load_large_data_from_pkl(
        '{}/{}'.format(dir_path, file_name),
        n_parts=len(indices), verbose=verbose)
  else:
    return load_data_from_pkl(
        '{}/{}.p'.format(dir_path, file_name),
        verbose=verbose)


def load_data_from_pkl(data_path, verbose=True):
  """Load data from pickle file."""
  with open(data_path, 'rb') as f:
    if verbose:
      print('Loading {}...'.format(f.name))
    return pickle.load(f)


def load_large_data_from_pkl(data_path, n_parts=2, verbose=True):
  """Save large data to pickle file."""
  if verbose:
    print('Loading {}.p from {} parts...'.format(data_path, n_parts))
  data = []
  for i in range(n_parts):
    with open(data_path + '_{}.p'.format(i), 'rb') as f:
      if verbose:
        print('Loading {}...'.format(f.name))
      data.append(pickle.load(f))
  concat = np.concatenate(data, axis=0)
  assert concat.shape[1:] == data[0].shape[1:]

  if verbose:
    print('Total Size: {:.4}Gb'.format(concat / (10**9)))
    print('Data Shape: ', concat.shape)

  return concat


def get_vec_length(vec, batch_size, epsilon):
  """Get the length of a vector."""
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
  """Check if directories exit or not."""
  for dir_path in path_list:
    if not isdir(dir_path):
      os.makedirs(dir_path)


def thin_line():
  print('-' * 55)


def thick_line():
  print('=' * 55)


def get_batches(x, y, batch_size, imgs=None):
  """Split features and labels into batches."""
  for start in range(0, len(x) - batch_size + 1, batch_size):
    end = start + batch_size
    if imgs:
      yield x[start:end], y[start:end], imgs[start:end]
    else:
      yield x[start:end], y[start:end]


def get_batches_all(x, y, batch_size):
  """Split features and labels into batches."""
  for start in range(0, len(x), batch_size):
    end = start + batch_size
    yield x[start:end], y[start:end]


def get_batches_all_x(x, batch_size):
  """Split features into batches."""
  for start in range(0, len(x), batch_size):
    end = start + batch_size
    yield x[start:end]


def print_status(epoch_i, epochs, step, start_time, loss_train,
                 clf_loss_train, rec_loss_train, acc_train, loss_valid,
                 clf_loss_valid, rec_loss_valid, acc_valid, with_rec):
  """Print information while training."""
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
  """Print information of full set evaluation."""
  thin_line()
  print('Epoch: {}/{} |'.format(epoch_i + 1, epochs),
        'Batch: {} |'.format(step),
        'Time: {:.2f}s |'.format(time.time() - start_time))
  thin_line()
  if with_full_set_eval:
    print('Full_Set_Train_Loss: {:.4f}'.format(loss_train))
    if with_rec:
      print('Train_Classifier_Loss: {:.4f}\n'.format(clf_loss_train),
            'Train_REC_LOSS: {:.4f}'.format(rec_loss_train))
    print('Full_Set_Train_Accuracy: {:.2f}%'.format(acc_train * 100))
  print('Full_Set_Valid_Loss: {:.4f}'.format(loss_valid))
  if with_rec:
    print('Valid_Classifier_Loss: {:.4f}\n'.format(clf_loss_valid),
          'Reconstruction_Valid_Loss: {:.4f}'.format(rec_loss_valid))
  print('Full_Set_Valid_Accuracy: {:.2f}%'.format(acc_valid * 100))


def print_multi_obj_eval(precision, recall, accuracy,
                         f1score, f05score, f2score):
  """Print information of multi-objects detection evaluation."""
  thin_line()
  print('Precision: {:.4f} \n'.format(precision),
        'Recall: {:.4f} \n'.format(recall),
        'Accuracy: {:.4f} \n'.format(accuracy),
        'F_1 Score: {:.4f} \n'.format(f1score),
        'F_0.5 Score: {:.4f} \n'.format(f05score),
        'F_2 Score: {:.4f} \n'.format(f2score),
        )


def save_config_log(file_path, cfg, clf_arch_info=None, rec_arch_info=None):
  """Save configuration of training."""
  file_path = os.path.join(file_path, 'config_log.txt')

  if not os.path.isfile(file_path):
    thick_line()
    print('Saving {}...'.format(file_path))
    with open(file_path, 'a') as f:
      local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                 time.localtime(time.time()))
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
          f.write(
              '[{}] {}\n\tParameters: {}\n\tOutput tensor shape: {}\n'.format(
                  i, clf_name, clf_params, clf_shape))
      if rec_arch_info is not None:
        f.write('=' * 55 + '\n')
        f.write('Reconstruction Architecture:\n')
        f.write('-' * 55 + '\n')
        for j, (rec_name, rec_params, rec_shape) in enumerate(rec_arch_info):
          f.write(
              '[{}] {}\n\tParameters: {}\n\tOutput tensor shape: {}\n'.format(
                  j, rec_name, rec_params, rec_shape))
      f.write('=' * 55)


def save_log(file_path, epoch_i, step, using_time,
             loss_train, clf_loss_train, rec_loss_train, acc_train,
             loss_valid, clf_loss_valid, rec_loss_valid, acc_valid, with_rec):
  """Save losses and accuracies while training."""
  if with_rec:
    if not os.path.isfile(file_path):
      with open(file_path, 'w') as f:
        header = ['Local_Time', 'Epoch', 'Batch', 'Time', 'Train_Loss',
                  'Train_Classifier_loss', 'Train_REC_LOSS',
                  'Train_Accuracy', 'Valid_Loss', 'Valid_Classifier_loss',
                  'Valid_REC_LOSS', 'Valid_Accuracy']
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
  """Save losses and accuracies of testing."""
  file_path = os.path.join(file_path, 'test_log.csv')
  thin_line()
  print('Saving {}...'.format(file_path))

  with open(file_path, 'a') as f:
    local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
    f.write('=' * 55 + '\n')
    f.write('Time: {}\n'.format(local_time))
    f.write('-' * 55 + '\n')
    f.write('Test Loss: {:.4f}\n'.format(loss_test))
    f.write('Test Accuracy: {:.2f}%\n'.format(acc_test * 100))
    if with_rec:
      f.write('Test Classifier Loss: {:.4f}\n'.format(clf_loss_test))
      f.write('Test Reconstruction Loss: {:.4f}\n'.format(rec_loss_test))
    f.write('=' * 55)


def save_test_log_is_training(file_path, epoch, step, loss_test, acc_test,
                              clf_loss_test, rec_loss_test, with_rec):
  """Save losses and accuracies of testing."""
  file_path = os.path.join(file_path, 'test_log.csv')

  if with_rec:
    if not os.path.isfile(file_path):
      with open(file_path, 'w') as f:
        header = ['Local_Time', 'Epoch', 'Batch', 'Test_Loss', 'Test_Accuracy']
        writer = csv.writer(f)
        writer.writerow(header)

    with open(file_path, 'a') as f:
      local_time = time.strftime(
          '%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
      log = [local_time, epoch, step, loss_test, acc_test]
      writer = csv.writer(f)
      writer.writerow(log)
  else:
    if not os.path.isfile(file_path):
      with open(file_path, 'w') as f:
        header = ['Local_Time', 'Epoch', 'Batch', 'Test_Loss',
                  'Test_Classifier_Loss', 'Test_Reconstruction_Loss',
                  'Test_Accuracy']
        writer = csv.writer(f)
        writer.writerow(header)

    with open(file_path, 'a') as f:
      local_time = time.strftime(
          '%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
      log = [local_time, epoch, step, loss_test,
             clf_loss_test, rec_loss_test, acc_test]
      writer = csv.writer(f)
      writer.writerow(log)


def save_multi_obj_scores(file_path, precision, recall,
                          accuracy, f1score, f05score, f2score):
  """Save evaluation scores of multi-objects detection."""
  file_path = os.path.join(file_path, 'multi_obj_scores.csv')
  thin_line()
  print('Saving {}...'.format(file_path))

  with open(file_path, 'a') as f:
    local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
    f.write('=' * 55 + '\n')
    f.write('Time: {}\n'.format(local_time))
    f.write('-' * 55 + '\n')
    f.write('Precision: {:.4f} \n'.format(precision))
    f.write('Recall: {:.4f} \n'.format(recall))
    f.write('Accuracy: {:.4f} \n'.format(accuracy))
    f.write('F_1 Score: {:.4f} \n'.format(f1score))
    f.write('F_0.5 Score: {:.4f} \n'.format(f05score))
    f.write('F_2 Score: {:.4f} \n'.format(f2score))
    f.write('=' * 55)


def save_multi_obj_scores_is_training(file_path, epoch, step, precision, recall,
                                      accuracy, f1score, f05score, f2score):
  """Save evaluation scores of multi-objects detection."""
  file_path = os.path.join(file_path, 'multi_obj_scores.csv')

  if not os.path.isfile(file_path):
    with open(file_path, 'w') as f:
      header = ['Local_Time', 'Epoch', 'Batch', 'Precision', 'Recall',
                'Accuracy', 'F_1 Score', 'F_0.5 Score', 'F_2 Score']
      writer = csv.writer(f)
      writer.writerow(header)

  with open(file_path, 'a') as f:
    local_time = time.strftime(
        '%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
    log = [local_time, epoch, step, precision, recall,
           accuracy, f1score, f05score, f2score]
    writer = csv.writer(f)
    writer.writerow(log)


def dummy_to_class(y):
  _class = []
  for y_i in y:
    y_idx = []
    for idx, y_ in enumerate(y_i):
      if y_ > 0:
        y_idx.append(idx)
    _class.append(y_idx)
  return _class


def save_test_pred(file_path, labels, preds, preds_vec,
                   save_num=None, pred_is_int=False):
  """Save predictions of multi-objects detection."""
  check_dir([file_path])
  file_path = os.path.join(file_path, 'pred_log.csv')
  thin_line()
  print('Saving {}...'.format(file_path))

  if save_num:
    save_idx = np.random.choice(len(labels), save_num, replace=False)
    labels, preds, preds_vec = \
        labels[save_idx], preds[save_idx], preds_vec[save_idx]

  if not pred_is_int:
    preds_class = dummy_to_class(preds)
  else:
    preds_class = preds
  labels_class = dummy_to_class(labels)

  if not os.path.isfile(file_path):
    with open(file_path, 'w') as f:
      header = ['sample_idx', 'labels_class', 'preds_class',
                'labels', 'preds', 'preds_vec']
      writer = csv.writer(f)
      writer.writerow(header)

  with open(file_path, 'a') as f:
    writer = csv.writer(f)
    for i in range(len(labels)):
      log = [i, labels_class[i], preds_class[i],
             labels[i], preds[i], preds_vec[i]]
      writer.writerow(log)


def save_test_pred_is_training(file_path, epoch, step, labels, preds, preds_vec,
                               save_num=None, pred_is_int=False):
  """Save predictions of multi-objects detection."""
  check_dir([file_path])
  file_path = os.path.join(file_path, 'pred_log.csv')
  thin_line()
  print('Saving {}...'.format(file_path))

  if save_num:
    save_idx = np.random.choice(len(labels), save_num, replace=False)
    labels, preds, preds_vec = \
        labels[save_idx], preds[save_idx], preds_vec[save_idx]

  if not pred_is_int:
    preds_class = dummy_to_class(preds)
  else:
    preds_class = preds
  labels_class = dummy_to_class(labels)

  if not os.path.isfile(file_path):
    with open(file_path, 'w') as f:
      header = ['Epoch', 'Batch', 'sample_idx', 'labels_class',
                'preds_class', 'labels', 'preds', 'preds_vec']
      writer = csv.writer(f)
      writer.writerow(header)

  with open(file_path, 'a') as f:
    writer = csv.writer(f)
    for i in range(len(labels)):
      log = [epoch, step, i, labels_class[i],
             preds_class[i], labels[i], preds[i], preds_vec[i]]
      writer.writerow(log)
    writer.writerow(['', '', '', '', '', '', '', ''])


def _read32(bytestream):
  """Read 32-bit integer from bytesteam.

  Args:
    bytestream: A bytestream

  Returns:
    32-bit integer
  """
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_image(save_path, extract_path):
  """Extract the images into a 4D unit8 numpy array [index, y, x, depth]."""
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
  """Extract the labels into a 1D uint8 numpy array [index]."""
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
  """Load a batch of the dataset."""
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


def img_add_overlap(imgs,
                    merge=False,
                    vec=None,
                    gamma=0):
  """Add images together with overlap."""
  if merge:
    c = 1 / len(imgs)
  else:
    c = 1
  added = np.zeros_like(imgs[0])
  for i, img in enumerate(imgs):
    if vec:
      added += img * (1 / vec[i]) * c
    else:
      added += img * c
  added += gamma
  added[added > 1] = 1
  added[added < 0] = 0
  return added


def img_add_no_overlap(imgs,
                       num_mul_obj,
                       vec=None,
                       img_mode='L',
                       resize_filter=None):
  """Add images together without overlap."""
  img_shape = np.array(imgs).shape[1:]
  save_size = \
      math.ceil(np.sqrt(num_mul_obj)) * img_shape[0]
  new_img = \
      np.zeros([save_size, save_size, img_shape[-1]]).astype('uint8')

  # Scale to 0-255
  imgs = imgs_scale_to_255(imgs)

  # Combine images
  row_start, row_end, col_start, col_end = 0, img_shape[0], 0, 0
  for i, img in enumerate(imgs):
    if vec:
      img = img * (1 / vec[i])
    col_end += img_shape[1]
    new_img[row_start:row_end, col_start:col_end, :] = img
    col_start += img_shape[1]
    if col_end == save_size:
      col_start = 0
      col_end = 0
      row_start += img_shape[0]
      row_end += img_shape[0]
  if img_mode == 'L':
      new_img = np.squeeze(new_img, axis=-1)
  added = Image.fromarray(new_img.astype('uint8'), mode=img_mode)
  added = np.expand_dims(
      added.resize(img_shape[:2], resize_filter), axis=-1) / 255.

  return added


def img_resize(imgs, img_shape, img_mode='L', resize_filter=None):
  """Resize images"""
  resized_imgs = []
  for img in tqdm(imgs, ncols=100, unit=' images'):
    if img_mode == 'L':
      img = np.squeeze(img, axis=-1)
    img = Image.fromarray(img.astype('uint8'), mode=img_mode)
    resized_img = np.expand_dims(
        img.resize(img_shape, resize_filter), axis=-1)
    resized_imgs.append(resized_img)
  return np.array(resized_imgs)


def img_black_to_color(imgs, same=False):
  color_coef_list = [[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 1, 1]]
  if same:
    # img shape: [n_test, 28, 28, 1] for mnist
    imgs_ = np.append(imgs, imgs, axis=3)
    imgs_ = np.append(imgs_, imgs, axis=3)
  else:
    # img shape: [n_y, 28, 28, 1] for mnist
    imgs_ = []
    for i, img in enumerate(imgs):
      img_colored = img * color_coef_list[i][0]
      img_colored = np.append(img_colored, img * color_coef_list[i][1], axis=2)
      img_colored = np.append(img_colored, img * color_coef_list[i][2], axis=2)
      imgs_.append(img_colored)
  return np.array(imgs_)


def imgs_scale_to_255(imgs):
  """Scale images to 0-255"""
  return np.array(
      [np.divide(((i - i.min()) * 255),
                 (i.max() - i.min())) for i in imgs]).astype(int)


def save_imgs(real_imgs,
              rec_imgs,
              img_path,
              database_name,
              max_img_in_col,
              step=None,
              silent=False,
              epoch_i=None,
              test_flag=False,
              colorful=False,
              append_info=None):
  """Save images to jpg files."""
  # Image shape
  img_shape = real_imgs.shape[1:]

  # Get maximum size for square grid of images
  save_col_size = math.floor(np.sqrt(rec_imgs.shape[0] * 2))
  if save_col_size > max_img_in_col:
    save_col_size = max_img_in_col
  save_row_size = save_col_size // 2

  # Scale to 0-255
  rec_images_ = imgs_scale_to_255(rec_imgs)
  real_images_ = imgs_scale_to_255(real_imgs)

  # Put images in a square arrangement
  rec_images_in_square = np.reshape(
      rec_images_[: save_row_size * save_col_size],
      (save_row_size, save_col_size, *img_shape)).astype(np.uint8)
  real_images_in_square = np.reshape(
      real_images_[: save_row_size * save_col_size],
      (save_row_size, save_col_size, *img_shape)).astype(np.uint8)

  mode = 'RGB'
  if database_name == 'mnist' or database_name == 'radical':
    if not colorful:
      mode = 'L'
      rec_images_in_square = np.squeeze(rec_images_in_square, 4)
      real_images_in_square = np.squeeze(real_images_in_square, 4)

  # Combine images to grid image
  thin_gap = 1
  thick_gap = 3
  avg_gap = (thin_gap + thick_gap) / 2
  new_im = Image.new(mode, (
    int((img_shape[1] + thin_gap) *
        save_col_size - thin_gap + thick_gap * 2),
    int((img_shape[0] + avg_gap) *
        save_row_size * 2 + thick_gap)), 'white')

  for row_i in range(save_row_size * 2):
    for col_i in range(save_col_size):
      if (row_i + 1) % 2 == 0:  # Odd
        if mode == 'L':
          image = rec_images_in_square[(row_i + 1) // 2 - 1, col_i, :, :]
        else:
          image = rec_images_in_square[(row_i + 1) // 2 - 1, col_i, :, :, :]
        im = Image.fromarray(image, mode)
        new_im.paste(im, (
          int(col_i * (img_shape[1] + thin_gap) + thick_gap),
          int(row_i * img_shape[0] + (row_i + 1) * avg_gap)))
      else:  # Even
        if mode == 'L':
          image = real_images_in_square[int((row_i + 1) // 2), col_i, :, :]
        else:
          image = real_images_in_square[int((row_i + 1) // 2), col_i, :, :, :]
        im = Image.fromarray(image, mode)
        new_im.paste(im, (
          int(col_i * (img_shape[1] + thin_gap) + thick_gap),
          int(row_i * (img_shape[0] + avg_gap) + thick_gap)))

  if test_flag:
    if step:
      save_image_path = join(img_path, 'test_batch_{}'.format(step))
    else:
      save_image_path = join(img_path, 'test')
  else:
    if epoch_i:
      save_image_path = join(
          img_path,
          'train_epoch_{}_batch_{}'.format(epoch_i, step))
    else:
      save_image_path = join(
          img_path, 'train_batch_{}'.format(step))

  save_image_path = save_image_path + '{}.jpg'.format(append_info)

  if not silent:
    thin_line()
    print('Saving image to {}...'.format(save_image_path))

  new_im.save(save_image_path)


def square_grid_show_imgs(images, mode=None):
  """Save images as a square grid."""
  # Get maximum size for square grid of images
  img_size = math.floor(np.sqrt(images.shape[0]))

  # Scale to 0-255
  images = (
        ((images - images.min()) * 255) / (images.max() - images.min())).astype(
    np.uint8)

  # Put images in a square arrangement
  images_in_square = np.reshape(
      images[:img_size * img_size],
      (img_size, img_size, images.shape[1], images.shape[2], images.shape[3]))

  # images_in_square.shape = (5, 5, 28, 28, 1)

  if mode == 'L':
    cmap = 'gray'
    images_in_square = np.squeeze(images_in_square, 4)
  else:
    cmap = None

  # Combine images to grid image
  new_im = Image.new(mode,
                     (images.shape[1] * img_size, images.shape[2] * img_size))
  for row_i, row_images in enumerate(images_in_square):
    for col_i, image in enumerate(row_images):
      im = Image.fromarray(image, mode)
      new_im.paste(im, (col_i * images.shape[1], row_i * images.shape[2]))

  plt.imshow(np.array(new_im), cmap=cmap)
  plt.show()


class DLProgress(tqdm):
  """Handle Progress Bar while Downloading."""
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
