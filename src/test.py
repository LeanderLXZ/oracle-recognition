from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
from os.path import join, isdir

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from config import config
from models import utils


class Test(object):

  def __init__(self, cfg):

    # Config
    self.cfg = cfg

    # Get checkpoint path
    self.checkpoint_path = join(
        cfg.CHECKPOINT_PATH,
        '{}/models.ckpt-{}'.format(self.cfg.TEST_VERSION, self.cfg.TEST_CKP_IDX))

    # Get log path, append information if the directory exist.
    test_log_path_ = join(
        self.cfg.TEST_LOG_PATH,
        '{}-{}'.format(self.cfg.TEST_VERSION, self.cfg.TEST_CKP_IDX))
    self.test_log_path = test_log_path_
    i_append_info = 0
    while isdir(self.test_log_path):
      i_append_info += 1
      self.test_log_path = test_log_path_ + '({})'.format(i_append_info)

    # Path for saving images
    self.test_image_path = join(self.test_log_path, 'images')

    # Check directory of paths
    utils.check_dir([self.test_log_path])
    if self.cfg.TEST_WITH_RECONSTRUCTION:
      if self.cfg.TEST_SAVE_IMAGE_STEP is not None:
        utils.check_dir([self.test_image_path])

    # Save config
    utils.save_config_log(self.test_log_path, self.cfg)

    # Load data
    utils.thick_line()
    print('Loading data...')
    utils.thin_line()
    preprocessed_path_ = join(cfg.DPP_DATA_PATH, cfg.DATABASE_NAME)
    self.x_test = utils.load_data_from_pkl(
        join(preprocessed_path_, 'x_test.p'))
    self.y_test = utils.load_data_from_pkl(
        join(preprocessed_path_, 'y_test.p'))

    # Calculate number of batches
    self.n_batch_test = len(self.y_test) // self.cfg.TEST_BATCH_SIZE

  def _get_tensors(self, loaded_graph):
    """
    Get inputs, labels, loss, and accuracy tensor from <loaded_graph>
    """
    with loaded_graph.as_default():

      utils.thin_line()
      print('Loading graph and tensors...')

      inputs_ = loaded_graph.get_tensor_by_name("inputs:0")
      labels_ = loaded_graph.get_tensor_by_name("labels:0")
      loss_ = loaded_graph.get_tensor_by_name("loss:0")
      accuracy_ = loaded_graph.get_tensor_by_name("accuracy:0")

      if self.cfg.TEST_WITH_RECONSTRUCTION:
        clf_loss_ = loaded_graph.get_tensor_by_name("classifier_loss:0")
        rec_loss_ = loaded_graph.get_tensor_by_name("rec_loss:0")
        rec_images_ = loaded_graph.get_tensor_by_name("rec_images:0")
        return inputs_, labels_, loss_, accuracy_, \
            clf_loss_, rec_loss_, rec_images_
      else:
        return inputs_, labels_, loss_, accuracy_

  def _save_images(self, sess, rec_images, inputs, labels,
                   x_batch, y_batch, step):
    """
    Save reconstructed images.
    """
    rec_images_ = sess.run(
        rec_images, feed_dict={inputs: x_batch, labels: y_batch})

    # Image shape
    img_shape = x_batch.shape[1:]

    # Get maximum size for square grid of images
    save_col_size = math.floor(np.sqrt(rec_images_.shape[0] * 2))
    if save_col_size > self.cfg.MAX_IMAGE_IN_COL:
      save_col_size = self.cfg.MAX_IMAGE_IN_COL
    save_row_size = save_col_size // 2

    # Scale to 0-255
    rec_images_ = np.array(
        [np.divide(((img_ - img_.min()) * 255), (img_.max() - img_.min()))
         for img_ in rec_images_])
    real_images_ = np.array(
        [np.divide(((img_ - img_.min()) * 255), (img_.max() - img_.min()))
         for img_ in x_batch])

    # Put images in a square arrangement
    rec_images_in_square = np.reshape(
        rec_images_[: save_row_size * save_col_size],
        (save_row_size, save_col_size, *img_shape)).astype(np.uint8)
    real_images_in_square = np.reshape(
        real_images_[: save_row_size*save_col_size],
        (save_row_size, save_col_size, *img_shape)).astype(np.uint8)

    if self.cfg.DATABASE_NAME == 'mnist':
      mode = 'L'
      rec_images_in_square = np.squeeze(rec_images_in_square, 4)
      real_images_in_square = np.squeeze(real_images_in_square, 4)
    else:
      mode = 'RGB'

    # Combine images to grid image
    thin_gap = 1
    thick_gap = 3
    avg_gap = (thin_gap + thick_gap) / 2
    new_im = Image.new(mode, (
        int((img_shape[1] + thin_gap)
            * save_col_size - thin_gap + thick_gap * 2),
        int((img_shape[0] + avg_gap)
            * save_row_size * 2 + thick_gap)), 'white')

    for row_i in range(save_row_size * 2):
      for col_i in range(save_col_size):
        if (row_i + 1) % 2 == 0:  # Odd
          if mode == 'L':
            image = rec_images_in_square[
                (row_i + 1) // 2 - 1, col_i, :, :]
          else:
            image = rec_images_in_square[
                (row_i + 1) // 2 - 1, col_i, :, :, :]
          im = Image.fromarray(image, mode)
          new_im.paste(im, (
              int(col_i * (img_shape[1] + thin_gap) + thick_gap),
              int(row_i * img_shape[0] + (row_i + 1) * avg_gap)))

        else:  # Even
          if mode == 'L':
            image = real_images_in_square[
                    int((row_i + 1) // 2), col_i, :, :]
          else:
            image = real_images_in_square[
                    int((row_i + 1) // 2), col_i, :, :, :]
          im = Image.fromarray(image, mode)
          new_im.paste(im, (
              int(col_i * (img_shape[1] + thin_gap) + thick_gap),
              int(row_i * (img_shape[0] + avg_gap) + thick_gap)))

    save_image_path = join(
        self.test_image_path, 'batch_{}.jpg'.format(step))
    new_im.save(save_image_path)

  def _eval_on_batches(self, sess, inputs, labels, loss, accuracy,
                       clf_loss, rec_loss, rec_images,  x, y, n_batch):
    """
    Calculate losses and accuracies of full train set.
    """
    loss_all = []
    acc_all = []
    clf_loss_all = []
    rec_loss_all = []
    step = 0
    _batch_generator = utils.get_batches(x, y, self.cfg.TEST_BATCH_SIZE)

    if self.cfg.TEST_WITH_RECONSTRUCTION:
      for _ in tqdm(range(n_batch), total=n_batch,
                    ncols=100, unit=' batches'):
        step += 1
        x_batch, y_batch = next(_batch_generator)
        loss_i, clf_loss_i, rec_loss_i, acc_i = \
            sess.run([loss, clf_loss, rec_loss, accuracy],
                     feed_dict={inputs: x_batch, labels: y_batch})
        loss_all.append(loss_i)
        clf_loss_all.append(clf_loss_i)
        rec_loss_all.append(rec_loss_i)
        acc_all.append(acc_i)

        # Save reconstruct images
        if self.cfg.TEST_SAVE_IMAGE_STEP is not None:
          if step % self.cfg.TEST_SAVE_IMAGE_STEP == 0:
            self._save_images(sess, rec_images, inputs, labels,
                              x_batch, y_batch, step)

      clf_loss = sum(clf_loss_all) / len(clf_loss_all)
      rec_loss = sum(rec_loss_all) / len(rec_loss_all)

    else:
      for _ in tqdm(range(n_batch), total=n_batch,
                    ncols=100, unit=' batches'):
        x_batch, y_batch = next(_batch_generator)
        loss_i, acc_i = \
            sess.run([loss, accuracy],
                     feed_dict={inputs: x_batch, labels: y_batch})
        loss_all.append(loss_i)
        acc_all.append(acc_i)
      clf_loss, rec_loss = None, None

    loss = sum(loss_all) / len(loss_all)
    accuracy = sum(acc_all) / len(acc_all)

    return loss, clf_loss, rec_loss, accuracy

  def test(self):
    """
    Test models
    """
    start_time = time.time()
    tf.reset_default_graph()
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:

      # Load saved models
      loader = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
      loader.restore(sess, self.checkpoint_path)

      # Get Tensors from loaded models
      if self.cfg.TEST_WITH_RECONSTRUCTION:
        inputs, labels, loss, accuracy, \
            clf_loss, rec_loss, rec_images = \
            self._get_tensors(loaded_graph)
      else:
        inputs, labels, loss, accuracy = self._get_tensors(loaded_graph)
        clf_loss, rec_loss, rec_images = None, None, None

      utils.thick_line()
      print('Testing on test set...')

      utils.thin_line()
      print('Calculating loss and accuracy of test set...')

      loss_test, clf_loss_test, rec_loss_test, acc_test = \
          self._eval_on_batches(
              sess, inputs, labels, loss, accuracy,
              clf_loss, rec_loss, rec_images,
              self.x_test, self.y_test, self.n_batch_test)

      # Print losses and accuracy
      utils.thin_line()
      print('Test_Loss: {:.4f}'.format(loss_test))
      if self.cfg.TEST_WITH_RECONSTRUCTION:
        print('Test_Classifier_Loss: {:.4f}\n'.format(clf_loss_test),
              'Test_Reconstruction_Loss: {:.4f}'.format(rec_loss_test))
      print('Test_Accuracy: {:.2f}%'.format(acc_test * 100))

      # Save test log
      utils.save_test_log(
          self.test_log_path, loss_test, acc_test, clf_loss_test,
          rec_loss_test, self.cfg.TEST_WITH_RECONSTRUCTION)

      utils.thin_line()
      print('Testing finished! Using time: {:.2f}'
            .format(time.time() - start_time))
      utils.thick_line()


if __name__ == '__main__':

  Test_ = Test(config)
  Test_.test()
