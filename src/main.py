from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from os import environ
from os.path import join, isdir

from config import config
from models.baseline_config import basel_config
from models import utils
from models.capsNet import CapsNet
from models.capsNet_distribute import CapsNetDistribute
from models.baseline_arch import basel_arch
from capsNet_arch import caps_arch


class Main(object):

  def __init__(self, model, cfg):
    """Load data and initialize models.

    Args:
      model: the models which will be trained
    """
    # Global start time
    self.start_time = time.time()

    # Config
    self.cfg = cfg

    # Get paths from configuration
    self.preprocessed_path, self.train_log_path, \
        self.test_log_path, self.summary_path, self.checkpoint_path, \
        self.train_image_path, self.test_image_path = self._get_paths()

    # Load data
    self.x_train, self.y_train, self.x_valid, self.y_valid = self._load_data()

    # Calculate number of batches
    self.n_batch_train = len(self.y_train) // cfg.BATCH_SIZE
    self.n_batch_valid = len(self.y_valid) // cfg.BATCH_SIZE

    # Build graph
    utils.thick_line()
    print('Building graph...')
    tf.reset_default_graph()
    self.step, self.train_graph, self.inputs, self.labels, self.is_training, \
        self.optimizer, self.saver, self.summary, self.loss, self.accuracy,\
        self.clf_loss, self.rec_loss, self.rec_images = model.build_graph(
            image_size=self.x_train.shape[1:],
            num_class=self.y_train.shape[1])

    # Save config
    utils.save_config_log(
        self.train_log_path, cfg, model.clf_arch_info, model.rec_arch_info)

  def _get_paths(self):
    """Get paths from configuration."""
    train_log_path_ = join(self.cfg.TRAIN_LOG_PATH, self.cfg.VERSION)
    test_log_path_ = join(self.cfg.TEST_LOG_PATH, self.cfg.VERSION)
    summary_path_ = join(self.cfg.SUMMARY_PATH, self.cfg.VERSION)
    checkpoint_path_ = join(self.cfg.CHECKPOINT_PATH, self.cfg.VERSION)
    preprocessed_path = join(self.cfg.DPP_DATA_PATH, self.cfg.DATABASE_NAME)

    # Get log paths, append information if the directory exist.
    train_log_path = train_log_path_
    i_append_info = 0
    while isdir(train_log_path):
      i_append_info += 1
      train_log_path = train_log_path_ + '({})'.format(i_append_info)

    if i_append_info > 0:
      summary_path = summary_path_ + '({})'.format(i_append_info)
      checkpoint_path = checkpoint_path_ + '({})'.format(i_append_info)
      test_log_path = test_log_path_ + '({})'.format(i_append_info)
    else:
      summary_path = summary_path_
      checkpoint_path = checkpoint_path_
      test_log_path = test_log_path_

    # Images saving path
    train_image_path = join(train_log_path, 'images')
    test_image_path = join(test_log_path, 'images')

    # Check directory of paths
    utils.check_dir([train_log_path, checkpoint_path])
    if self.cfg.WITH_RECONSTRUCTION:
      if self.cfg.SAVE_IMAGE_STEP:
        utils.check_dir([train_image_path])

    return preprocessed_path, train_log_path, test_log_path, \
        summary_path, checkpoint_path, train_image_path, test_image_path

  def _load_data(self):
    """Load preprocessed data."""
    utils.thick_line()
    print('Loading data...')
    utils.thin_line()

    # if self.cfg.DATABASE_NAME == 'radical':
    #   self.x_train = utils.load_large_data_to_pkl(
    #       join(self.preprocessed_path, 'x_train'),
    #       n_parts=self.cfg.LARGE_DATA_PART_NUM)
    # else:
    x_train = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'x_train.p'))
    y_train = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'y_train.p'))
    x_valid = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'x_valid.p'))
    y_valid = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'y_valid.p'))

    utils.thin_line()
    print('Data info:')
    utils.thin_line()
    print('x_train: {}\ny_train: {}\nx_valid: {}\ny_valid: {}'.format(
        x_train.shape,
        y_train.shape,
        x_valid.shape,
        y_valid.shape))

    return x_train, y_train, x_valid, y_valid

  def _display_status(self,
                      sess,
                      x_batch,
                      y_batch,
                      epoch_i,
                      step):
    """Display information during training."""
    valid_batch_idx = np.random.choice(
        range(len(self.x_valid)), self.cfg.BATCH_SIZE).tolist()
    x_valid_batch = self.x_valid[valid_batch_idx]
    y_valid_batch = self.y_valid[valid_batch_idx]

    if self.cfg.WITH_RECONSTRUCTION:
      loss_train, clf_loss_train, rec_loss_train, acc_train = \
          sess.run([self.loss, self.clf_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch,
                              self.is_training: False})
      loss_valid, clf_loss_valid, rec_loss_valid, acc_valid = \
          sess.run([self.loss, self.clf_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch,
                              self.is_training: False})
    else:
      loss_train, acc_train = \
          sess.run([self.loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch,
                              self.is_training: False})
      loss_valid, acc_valid = \
          sess.run([self.loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch,
                              self.is_training: False})
      clf_loss_train, rec_loss_train, clf_loss_valid, rec_loss_valid = \
          None, None, None, None

    utils.print_status(
        epoch_i, self.cfg.EPOCHS, step, self.start_time,
        loss_train, clf_loss_train, rec_loss_train, acc_train,
        loss_valid, clf_loss_valid, rec_loss_valid, acc_valid,
        self.cfg.WITH_RECONSTRUCTION)

  def _save_logs(self,
                 sess,
                 train_writer,
                 valid_writer,
                 x_batch,
                 y_batch,
                 epoch_i,
                 step):
    """Save logs and ddd summaries to TensorBoard while training."""
    valid_batch_idx = np.random.choice(
        range(len(self.x_valid)), self.cfg.BATCH_SIZE).tolist()
    x_valid_batch = self.x_valid[valid_batch_idx]
    y_valid_batch = self.y_valid[valid_batch_idx]

    if self.cfg.WITH_RECONSTRUCTION:
      summary_train, loss_train, clf_loss_train, rec_loss_train, acc_train = \
          sess.run([self.summary, self.loss, self.clf_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch,
                              self.is_training: False})
      summary_valid, loss_valid, clf_loss_valid, rec_loss_valid, acc_valid = \
          sess.run([self.summary, self.loss, self.clf_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch,
                              self.is_training: False})
    else:
      summary_train, loss_train, acc_train = \
          sess.run([self.summary, self.loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch,
                              self.is_training: False})
      summary_valid, loss_valid, acc_valid = \
          sess.run([self.summary, self.loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch,
                              self.is_training: False})
      clf_loss_train, rec_loss_train, clf_loss_valid, rec_loss_valid = \
          None, None, None, None

    train_writer.add_summary(summary_train, step)
    valid_writer.add_summary(summary_valid, step)
    utils.save_log(
        join(self.train_log_path, 'train_log.csv'), epoch_i + 1, step,
        time.time() - self.start_time, loss_train, clf_loss_train,
        rec_loss_train, acc_train, loss_valid, clf_loss_valid, rec_loss_valid,
        acc_valid, self.cfg.WITH_RECONSTRUCTION)

  def _eval_on_batches(self,
                       mode,
                       sess,
                       x,
                       y,
                       n_batch,
                       silent=False):
    """Calculate losses and accuracies of full train set."""
    loss_all = []
    acc_all = []
    clf_loss_all = []
    rec_loss_all = []

    if not silent:
      utils.thin_line()
      print('Calculating loss and accuracy of full {} set...'.format(mode))
      _batch_generator = utils.get_batches(x, y, self.cfg.BATCH_SIZE)

      if self.cfg.WITH_RECONSTRUCTION:
        for _ in tqdm(range(n_batch), total=n_batch,
                      ncols=100, unit=' batches'):
          x_batch, y_batch = next(_batch_generator)
          loss_i, clf_loss_i, rec_loss_i, acc_i = sess.run(
              [self.loss, self.clf_loss, self.rec_loss, self.accuracy],
              feed_dict={self.inputs: x_batch,
                         self.labels: y_batch,
                         self.is_training: False})
          loss_all.append(loss_i)
          clf_loss_all.append(clf_loss_i)
          rec_loss_all.append(rec_loss_i)
          acc_all.append(acc_i)
        clf_loss = sum(clf_loss_all) / len(clf_loss_all)
        rec_loss = sum(rec_loss_all) / len(rec_loss_all)
      else:
        for _ in tqdm(range(n_batch), total=n_batch,
                      ncols=100, unit=' batches'):
          x_batch, y_batch = next(_batch_generator)
          loss_i, acc_i = sess.run(
              [self.loss, self.accuracy],
              feed_dict={self.inputs: x_batch,
                         self.labels: y_batch,
                         self.is_training: False})
          loss_all.append(loss_i)
          acc_all.append(acc_i)
        clf_loss, rec_loss = None, None

    else:
      if self.cfg.WITH_RECONSTRUCTION:
        for x_batch, y_batch in utils.get_batches(x, y, self.cfg.BATCH_SIZE):
          loss_i, clf_loss_i, rec_loss_i, acc_i = sess.run(
              [self.loss, self.clf_loss, self.rec_loss, self.accuracy],
              feed_dict={self.inputs: x_batch,
                         self.labels: y_batch,
                         self.is_training: False})
          loss_all.append(loss_i)
          clf_loss_all.append(clf_loss_i)
          rec_loss_all.append(rec_loss_i)
          acc_all.append(acc_i)
        clf_loss = sum(clf_loss_all) / len(clf_loss_all)
        rec_loss = sum(rec_loss_all) / len(rec_loss_all)
      else:
        for x_batch, y_batch in utils.get_batches(x, y, self.cfg.BATCH_SIZE):
          loss_i, acc_i = sess.run(
              [self.loss, self.accuracy],
              feed_dict={self.inputs: x_batch,
                         self.labels: y_batch,
                         self.is_training: False})
          loss_all.append(loss_i)
          acc_all.append(acc_i)
        clf_loss, rec_loss = None, None

    loss = sum(loss_all) / len(loss_all)
    accuracy = sum(acc_all) / len(acc_all)

    return loss, clf_loss, rec_loss, accuracy

  def _eval_on_full_set(self,
                        sess,
                        epoch_i,
                        step,
                        silent=False):
    """Evaluate on the full data set and print information."""
    eval_start_time = time.time()

    if not silent:
      utils.thick_line()
      print('Calculating losses using full data set...')

    # Calculate losses and accuracies of full train set
    if self.cfg.EVAL_WITH_FULL_TRAIN_SET:
      loss_train, clf_loss_train, rec_loss_train, acc_train = \
          self._eval_on_batches('train', sess, self.x_train, self.y_train,
                                self.n_batch_train, silent=silent)
    else:
      loss_train, clf_loss_train, rec_loss_train, acc_train = \
          None, None, None, None

    # Calculate losses and accuracies of full valid set
    loss_valid, clf_loss_valid, rec_loss_valid, acc_valid = \
        self._eval_on_batches('valid', sess, self.x_valid, self.y_valid,
                              self.n_batch_valid, silent=silent)

    if not silent:
      utils.print_full_set_eval(
          epoch_i, self.cfg.EPOCHS, step, self.start_time,
          loss_train, clf_loss_train, rec_loss_train, acc_train,
          loss_valid, clf_loss_valid, rec_loss_valid, acc_valid,
          self.cfg.EVAL_WITH_FULL_TRAIN_SET, self.cfg.WITH_RECONSTRUCTION)

    file_path = join(self.train_log_path, 'full_set_eval_log.csv')
    if not silent:
      utils.thin_line()
      print('Saving {}...'.format(file_path))
    utils.save_log(
      file_path, epoch_i + 1, step, time.time() - self.start_time,
      loss_train, clf_loss_train, rec_loss_train, acc_train,
      loss_valid, clf_loss_valid, rec_loss_valid, acc_valid,
      self.cfg.WITH_RECONSTRUCTION)

    if not silent:
      utils.thin_line()
      print('Evaluation done! Using time: {:.2f}'
            .format(time.time() - eval_start_time))

  def _save_images(self,
                   sess,
                   img_path,
                   x,
                   y,
                   step,
                   silent=False,
                   epoch_i=None,
                   test_flag=False):
    """Save reconstructed images."""
    rec_images_ = sess.run(
        self.rec_images, feed_dict={self.inputs: x,
                                    self.labels: y,
                                    self.is_training: False})

    # rec_images_ shape: [128, 28, 28, 1] for mnist
    utils.save_imgs(
        real_imgs=x,
        rec_imgs=rec_images_,
        img_path=img_path,
        database_name=self.cfg.DATABASE_NAME,
        max_img_in_col=self.cfg.MAX_IMAGE_IN_COL,
        step=step,
        silent=silent,
        epoch_i=epoch_i,
        test_flag=test_flag)

  def _save_model(self, sess, saver, step, silent=False):
    """Save models."""
    save_path = join(self.checkpoint_path, 'models.ckpt')
    if not silent:
      utils.thin_line()
      print('Saving models to {}...'.format(save_path))
    saver.save(sess, save_path, global_step=step)

  def _test_after_training(self, sess):
    """Evaluate on the test set after training."""
    test_start_time = time.time()

    utils.thick_line()
    print('Testing...')

    # Check directory of paths
    utils.check_dir([self.test_log_path])
    if self.cfg.WITH_RECONSTRUCTION:
      if self.cfg.TEST_SAVE_IMAGE_STEP:
        utils.check_dir([self.test_image_path])

    # Load data
    utils.thin_line()
    print('Loading test set...')
    utils.thin_line()
    x_test = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'x_test.p'))
    y_test = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'y_test.p'))
    n_batch_test = len(y_test) // self.cfg.BATCH_SIZE

    utils.thin_line()
    print('Calculating loss and accuracy on test set...')
    loss_test_all = []
    acc_test_all = []
    clf_loss_test_all = []
    rec_loss_test_all = []
    step = 0
    _test_batch_generator = utils.get_batches(
        x_test, y_test, self.cfg.BATCH_SIZE)

    if self.cfg.WITH_RECONSTRUCTION:
      for _ in tqdm(range(n_batch_test), total=n_batch_test,
                    ncols=100, unit=' batches'):
        step += 1
        test_batch_x, test_batch_y = next(_test_batch_generator)
        loss_test_i, clf_loss_i, rec_loss_i, acc_test_i = sess.run(
            [self.loss, self.clf_loss, self.rec_loss, self.accuracy],
            feed_dict={self.inputs: test_batch_x,
                       self.labels: test_batch_y,
                       self.is_training: False})
        loss_test_all.append(loss_test_i)
        acc_test_all.append(acc_test_i)
        clf_loss_test_all.append(clf_loss_i)
        rec_loss_test_all.append(rec_loss_i)

        # Save reconstruct images
        if self.cfg.TEST_SAVE_IMAGE_STEP:
          if step % self.cfg.TEST_SAVE_IMAGE_STEP == 0:
            self._save_images(
                sess, self.test_image_path, test_batch_x,
                test_batch_y, step, silent=True, test_flag=True)

      clf_loss_test = sum(clf_loss_test_all) / len(clf_loss_test_all)
      rec_loss_test = sum(rec_loss_test_all) / len(rec_loss_test_all)

    else:
      for _ in tqdm(range(n_batch_test), total=n_batch_test,
                    ncols=100, unit=' batches'):
        test_batch_x, test_batch_y = next(_test_batch_generator)
        loss_test_i, acc_test_i = sess.run(
            [self.loss, self.accuracy],
            feed_dict={self.inputs: test_batch_x,
                       self.labels: test_batch_y,
                       self.is_training: False})
        loss_test_all.append(loss_test_i)
        acc_test_all.append(acc_test_i)
      clf_loss_test, rec_loss_test = None, None

    loss_test = sum(loss_test_all) / len(loss_test_all)
    acc_test = sum(acc_test_all) / len(acc_test_all)

    # Print losses and accuracy
    utils.thin_line()
    print('Test_Loss: {:.4f}\n'.format(loss_test),
          'Test_Accuracy: {:.2f}%'.format(acc_test * 100))
    if self.cfg.WITH_RECONSTRUCTION:
      utils.thin_line()
      print('Test_Train_Loss: {:.4f}\n'.format(clf_loss_test),
            'Test_Reconstruction_Loss: {:.4f}'.format(rec_loss_test))

    # Save test log
    utils.save_test_log(
        self.test_log_path, loss_test, acc_test, clf_loss_test,
        rec_loss_test, self.cfg.WITH_RECONSTRUCTION)

    utils.thin_line()
    print('Testing finished! Using time: {:.2f}'
          .format(time.time() - test_start_time))

  def _trainer(self, sess):

    utils.thick_line()
    print('Training...')

    # Merge all the summaries and create writers
    train_summary_path = join(self.summary_path, 'train')
    valid_summary_path = join(self.summary_path, 'valid')
    utils.check_dir([train_summary_path, valid_summary_path])
    train_writer = tf.summary.FileWriter(train_summary_path, sess.graph)
    valid_writer = tf.summary.FileWriter(valid_summary_path)

    sess.run(tf.global_variables_initializer())
    step = 0

    for epoch_i in range(self.cfg.EPOCHS):

      epoch_start_time = time.time()
      utils.thick_line()
      print('Training on epoch: {}/{}'.format(epoch_i + 1, self.cfg.EPOCHS))

      if self.cfg.DISPLAY_STEP:

        for x_batch, y_batch in utils.get_batches(self.x_train,
                                                  self.y_train,
                                                  self.cfg.BATCH_SIZE):
          step += 1

          # Training optimizer
          sess.run(self.optimizer, feed_dict={self.inputs: x_batch,
                                              self.labels: y_batch,
                                              self.step: step-1,
                                              self.is_training: True})

          # Display training information
          if step % self.cfg.DISPLAY_STEP == 0:
            self._display_status(sess, x_batch, y_batch, epoch_i, step-1)

          # Save training logs
          if self.cfg.SAVE_LOG_STEP:
            if step % self.cfg.SAVE_LOG_STEP == 0:
              self._save_logs(sess, train_writer, valid_writer,
                              x_batch, y_batch, epoch_i, step-1)

          # Save reconstruction images
          if self.cfg.SAVE_IMAGE_STEP:
            if self.cfg.WITH_RECONSTRUCTION:
              if step % self.cfg.SAVE_IMAGE_STEP == 0:
                self._save_images(
                    sess, self.train_image_path, x_batch,
                    y_batch, step-1, epoch_i=epoch_i)

          # Save models
          if self.cfg.SAVE_MODEL_MODE == 'per_batch':
            if step % self.cfg.SAVE_MODEL_STEP == 0:
              self._save_model(sess, self.saver, step-1)

          # Evaluate on full set
          if self.cfg.FULL_SET_EVAL_MODE == 'per_batch':
            if step % self.cfg.FULL_SET_EVAL_STEP == 0:
              self._eval_on_full_set(sess, epoch_i, step-1)
              utils.thick_line()
      else:
        utils.thin_line()
        train_batch_generator = utils.get_batches(
            self.x_train, self.y_train, self.cfg.BATCH_SIZE)
        for _ in tqdm(range(self.n_batch_train),
                      total=self.n_batch_train,
                      ncols=100, unit=' batches'):

          step += 1
          x_batch, y_batch = next(train_batch_generator)

          # Training optimizer
          sess.run(self.optimizer, feed_dict={self.inputs: x_batch,
                                              self.labels: y_batch,
                                              self.step: step-1,
                                              self.is_training: True})

          # Save training logs
          if self.cfg.SAVE_LOG_STEP:
            if step % self.cfg.SAVE_LOG_STEP == 0:
              self._save_logs(sess, train_writer, valid_writer,
                              x_batch, y_batch, epoch_i, step-1)

          # Save reconstruction images
          if self.cfg.SAVE_IMAGE_STEP:
            if self.cfg.WITH_RECONSTRUCTION:
              if step % self.cfg.SAVE_IMAGE_STEP == 0:
                self._save_images(
                    sess, self.train_image_path, x_batch,
                    y_batch, step-1, silent=True, epoch_i=epoch_i)

          # Save models per batch
          if self.cfg.SAVE_MODEL_MODE == 'per_batch':
            if step % self.cfg.SAVE_MODEL_STEP == 0:
              self._save_model(sess, self.saver, step-1, silent=True)

          # Evaluate on full set
          if self.cfg.FULL_SET_EVAL_MODE == 'per_batch':
            if step % self.cfg.FULL_SET_EVAL_STEP == 0:
              self._eval_on_full_set(sess, epoch_i, step-1, silent=True)

      # Save model per epoch
      if self.cfg.SAVE_MODEL_MODE == 'per_epoch':
        if (epoch_i + 1) % self.cfg.SAVE_MODEL_STEP == 0:
          self._save_model(sess, self.saver, epoch_i)

      # Evaluate per epoch
      if self.cfg.FULL_SET_EVAL_MODE == 'per_epoch':
        if (epoch_i + 1) % self.cfg.FULL_SET_EVAL_STEP == 0:
          self._eval_on_full_set(sess, epoch_i, step-1)

      utils.thin_line()
      print('Epoch {}/{} done! Using time: {:.2f}'
            .format(epoch_i + 1, self.cfg.EPOCHS,
                    time.time() - epoch_start_time))

    utils.thick_line()
    print('Training finished! Using time: {:.2f}'
          .format(time.time() - self.start_time))
    utils.thick_line()

    # Evaluate on test set after training
    if self.cfg.TEST_AFTER_TRAINING:
      self._test_after_training(sess)

    utils.thick_line()
    print('All task finished! Total time: {:.2f}'
          .format(time.time() - self.start_time))
    utils.thick_line()

  def train(self):
    """Training models."""
    session_cfg = tf.ConfigProto(allow_soft_placement=True)
    session_cfg.gpu_options.allow_growth = True

    if self.cfg.VAR_ON_CPU:
      with tf.Session(graph=self.train_graph, config=session_cfg) as sess:
        with tf.device('/cpu:0'):
          self._trainer(sess)
    else:
      with tf.Session(graph=self.train_graph, config=session_cfg) as sess:
        self._trainer(sess)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(
    description="Training the model."
  )
  parser.add_argument('-g', '--gpu', nargs="+",
                      choices=[0, 1], type=int, metavar='',
                      help="Run single-gpu version."
                           "Choose the GPU from: {!s}".format([0, 1]))
  parser.add_argument('-m', '--multi', action="store_true",
                      help="Run multi-gpu version.")
  parser.add_argument('-b', '--baseline', action="store_true",
                      help="Using baseline architecture and parameters.")
  args = parser.parse_args()

  if args.gpu:
    utils.thick_line()
    print('Using /gpu: %d' % args.gpu)
    environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    CapsNet_ = CapsNet
  elif args.multi:
    utils.thick_line()
    print('Running multi-gpu version.')
    CapsNet_ = CapsNetDistribute
  else:
    utils.thick_line()
    print('Input [ 1 ] to run normal version.')
    print('Input [ 2 ] to run multi-gpu version.')
    utils.thin_line()
    input_ = input('Input: ')
    if input_ == '1':
      CapsNet_ = CapsNet
    elif input_ == '2':
      CapsNet_ = CapsNetDistribute
    else:
      raise ValueError('Wrong Input! Found: ', input_)

  if args.baseline:
    print('Running baseline model.')
    arch_ = basel_arch
    config_ = basel_config
  else:
    arch_ = caps_arch
    config_ = config

  Main(CapsNet_(config_, arch_), config_).train()
