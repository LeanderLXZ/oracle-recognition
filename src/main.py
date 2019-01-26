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
from baseline_config import config as basel_cfg
from models import utils
from models.capsNet import CapsNet
from models.capsNet_distribute import CapsNetDistribute
from models.capsNet_multi_tasks import CapsNetMultiTasks
from baseline_arch import basel_arch
from capsNet_arch import caps_arch
from test import Test, TestMultiObjects, TestOracle


class Main(object):

  def __init__(self, cfg, model_arch, mode='normal'):
    """Load data and initialize models."""
    # Global start time
    self.start_time = time.time()

    # Config
    self.cfg = cfg
    self.multi_gpu = True

    if mode == 'multi-tasks':
      self.multi_gpu = True
      model = CapsNetMultiTasks(cfg, model_arch)
    elif mode == 'multi-gpu':
      self.multi_gpu = True
      model = CapsNetDistribute(cfg, model_arch)
    else:
      self.multi_gpu = False
      model = CapsNet(cfg, model_arch)

    # Use encode transfer learning
    if self.cfg.TRANSFER_LEARNING == 'encode':
      self.tl_encode = True
    else:
      self.tl_encode = False

    # Get paths from configuration
    self.preprocessed_path, self.train_log_path, \
        self.summary_path, self.checkpoint_path, \
        self.train_image_path = self._get_paths()

    # Load data
    self.x_train, self.y_train, self.x_valid, \
        self.y_valid, self.imgs_train, self.imgs_valid = self._load_data()

    # Calculate number of batches
    self.n_batch_train = len(self.y_train) // cfg.BATCH_SIZE
    self.n_batch_valid = len(self.y_valid) // cfg.BATCH_SIZE

    # Build graph
    utils.thick_line()
    print('Building graph...')
    tf.reset_default_graph()
    self.step, self.train_graph, self.inputs, self.labels, self.input_imgs,\
        self.is_training, self.optimizer, self.saver, self.summary, \
        self.loss, self.accuracy, self.clf_loss, self.rec_loss, \
        self.rec_images, self.preds = model.build_graph(
            input_size=self.x_train.shape[1:],
            image_size=self.imgs_train[1:],
            num_class=self.y_train.shape[1])

    # Save config
    self.clf_arch_info = model.clf_arch_info
    self.rec_arch_info = model.rec_arch_info
    utils.save_config_log(
        self.train_log_path, cfg, self.clf_arch_info, self.rec_arch_info)

  def _get_paths(self):
    """Get paths from configuration."""
    train_log_path_ = join(self.cfg.TRAIN_LOG_PATH, self.cfg.VERSION)
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
    else:
      summary_path = summary_path_
      checkpoint_path = checkpoint_path_

    # Images saving path
    train_image_path = join(train_log_path, 'images')

    # Check directory of paths
    utils.check_dir([train_log_path, checkpoint_path])
    if self.cfg.WITH_REC:
      if self.cfg.SAVE_IMAGE_STEP:
        utils.check_dir([train_image_path])

    return preprocessed_path, train_log_path, \
        summary_path, checkpoint_path, train_image_path

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
    x_valid = utils.load_data_from_pkl(
      join(self.preprocessed_path, 'x_train.p'))

    imgs_train = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'imgs_train.p'))
    imgs_valid = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'imgs_valid.p'))

    y_train = utils.load_data_from_pkl(
        join(self.preprocessed_path, 'y_train.p'))
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

    print('imgs_train: {}\nimgs_valid: {}'.format(
        imgs_train.shape,
        imgs_valid.shape))

    return x_train, y_train, x_valid, y_valid, imgs_train, imgs_valid

  def _display_status(self,
                      sess,
                      x_batch,
                      y_batch,
                      imgs_batch,
                      epoch_i,
                      step):
    """Display information during training."""
    valid_batch_idx = np.random.choice(
        range(len(self.x_valid)), self.cfg.BATCH_SIZE).tolist()
    x_valid_batch = self.x_valid[valid_batch_idx]
    y_valid_batch = self.y_valid[valid_batch_idx]
    imgs_valid_batch = self.imgs_valid[valid_batch_idx]

    if self.cfg.WITH_REC:
      loss_train, clf_loss_train, rec_loss_train, acc_train = \
          sess.run([self.loss, self.clf_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch,
                              self.input_imgs: imgs_batch,
                              self.is_training: False})
      loss_valid, clf_loss_valid, rec_loss_valid, acc_valid = \
          sess.run([self.loss, self.clf_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch,
                              self.input_imgs: imgs_valid_batch,
                              self.is_training: False})
    else:
      loss_train, acc_train = \
          sess.run([self.loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch,
                              self.input_imgs: imgs_batch,
                              self.is_training: False})
      loss_valid, acc_valid = \
          sess.run([self.loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch,
                              self.input_imgs: imgs_valid_batch,
                              self.is_training: False})
      clf_loss_train, rec_loss_train, clf_loss_valid, rec_loss_valid = \
          None, None, None, None

    utils.print_status(
        epoch_i, self.cfg.EPOCHS, step, self.start_time,
        loss_train, clf_loss_train, rec_loss_train, acc_train,
        loss_valid, clf_loss_valid, rec_loss_valid, acc_valid,
        self.cfg.WITH_REC)

  def _save_logs(self,
                 sess,
                 train_writer,
                 valid_writer,
                 x_batch,
                 y_batch,
                 imgs_batch,
                 epoch_i,
                 step):
    """Save logs and ddd summaries to TensorBoard while training."""
    valid_batch_idx = np.random.choice(
        range(len(self.x_valid)), self.cfg.BATCH_SIZE).tolist()
    x_valid_batch = self.x_valid[valid_batch_idx]
    y_valid_batch = self.y_valid[valid_batch_idx]
    imgs_valid_batch = self.imgs_valid[valid_batch_idx]

    if self.cfg.WITH_REC:
      summary_train, loss_train, clf_loss_train, rec_loss_train, acc_train = \
          sess.run([self.summary, self.loss, self.clf_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch,
                              self.input_imgs: imgs_batch,
                              self.is_training: False})
      summary_valid, loss_valid, clf_loss_valid, rec_loss_valid, acc_valid = \
          sess.run([self.summary, self.loss, self.clf_loss,
                    self.rec_loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch,
                              self.input_imgs: imgs_valid_batch,
                              self.is_training: False})
    else:
      summary_train, loss_train, acc_train = \
          sess.run([self.summary, self.loss, self.accuracy],
                   feed_dict={self.inputs: x_batch,
                              self.labels: y_batch,
                              self.input_imgs: imgs_batch,
                              self.is_training: False})
      summary_valid, loss_valid, acc_valid = \
          sess.run([self.summary, self.loss, self.accuracy],
                   feed_dict={self.inputs: x_valid_batch,
                              self.labels: y_valid_batch,
                              self.input_imgs: imgs_valid_batch,
                              self.is_training: False})
      clf_loss_train, rec_loss_train, clf_loss_valid, rec_loss_valid = \
          None, None, None, None

    train_writer.add_summary(summary_train, step)
    valid_writer.add_summary(summary_valid, step)
    utils.save_log(
        join(self.train_log_path, 'train_log.csv'), epoch_i + 1, step,
        time.time() - self.start_time, loss_train, clf_loss_train,
        rec_loss_train, acc_train, loss_valid, clf_loss_valid, rec_loss_valid,
        acc_valid, self.cfg.WITH_REC)

  def _eval_on_batches(self,
                       mode,
                       sess,
                       x,
                       y,
                       imgs,
                       n_batch,
                       silent=False):
    """Calculate losses and accuracies of full train set."""
    loss_all = []
    acc_all = []
    clf_loss_all = []
    rec_loss_all = []

    batch_generator = utils.get_batches(x, y, self.cfg.BATCH_SIZE, imgs=imgs)

    if not silent:
      utils.thin_line()
      print('Calculating loss and accuracy of full {} set...'.format(mode))
      iterator = tqdm(range(n_batch), total=n_batch, ncols=100, unit=' batches')
    else:
      iterator = range(n_batch)

    if self.cfg.WITH_REC:
      for _ in iterator:

        x_batch, y_batch, imgs_batch = next(batch_generator)

        loss_i, clf_loss_i, rec_loss_i, acc_i = sess.run(
            [self.loss, self.clf_loss, self.rec_loss, self.accuracy],
            feed_dict={self.inputs: x_batch,
                       self.labels: y_batch,
                       self.input_imgs: imgs_batch,
                       self.is_training: False})
        loss_all.append(loss_i)
        clf_loss_all.append(clf_loss_i)
        rec_loss_all.append(rec_loss_i)
        acc_all.append(acc_i)
      clf_loss = sum(clf_loss_all) / len(clf_loss_all)
      rec_loss = sum(rec_loss_all) / len(rec_loss_all)
    else:
      for _ in iterator:

        x_batch, y_batch, imgs_batch = next(batch_generator)

        loss_i, acc_i = sess.run(
            [self.loss, self.accuracy],
            feed_dict={self.inputs: x_batch,
                       self.labels: y_batch,
                       self.input_imgs: imgs_batch,
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
          self._eval_on_batches(
              'train', sess, self.x_train, self.y_train,
              self.imgs_train, self.n_batch_train, silent=silent)
    else:
      loss_train, clf_loss_train, rec_loss_train, acc_train = \
          None, None, None, None

    # Calculate losses and accuracies of full valid set
    loss_valid, clf_loss_valid, rec_loss_valid, acc_valid = \
        self._eval_on_batches(
            'valid', sess, self.x_valid, self.y_valid,
            self.imgs_valid, self.n_batch_valid, silent=silent)

    if not silent:
      utils.print_full_set_eval(
          epoch_i, self.cfg.EPOCHS, step, self.start_time,
          loss_train, clf_loss_train, rec_loss_train, acc_train,
          loss_valid, clf_loss_valid, rec_loss_valid, acc_valid,
          self.cfg.EVAL_WITH_FULL_TRAIN_SET, self.cfg.WITH_REC)

    file_path = join(self.train_log_path, 'full_set_eval_log.csv')
    if not silent:
      utils.thin_line()
      print('Saving {}...'.format(file_path))
    utils.save_log(
      file_path, epoch_i + 1, step, time.time() - self.start_time,
      loss_train, clf_loss_train, rec_loss_train, acc_train,
      loss_valid, clf_loss_valid, rec_loss_valid, acc_valid,
      self.cfg.WITH_REC)

    if not silent:
      utils.thin_line()
      print('Evaluation done! Using time: {:.2f}'
            .format(time.time() - eval_start_time))

  def _save_images(self,
                   sess,
                   img_path,
                   x,
                   y,
                   imgs,
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
        real_imgs=imgs,
        rec_imgs=rec_images_,
        img_path=img_path,
        database_name=self.cfg.DATABASE_NAME,
        max_img_in_col=self.cfg.MAX_IMAGE_IN_COL,
        step=step,
        silent=silent,
        epoch_i=epoch_i,
        test_flag=test_flag)

  def _save_model(self,
                  sess,
                  saver,
                  step,
                  silent=False):
    """Save models."""
    save_path = join(self.checkpoint_path, 'models.ckpt')
    if not silent:
      utils.thin_line()
      print('Saving models to {}...'.format(save_path))
    saver.save(sess, save_path, global_step=step)

  def _test(self,
            sess,
            is_training=False,
            epoch=None,
            step=None,
            mode='single'):
    """Evaluate on the test set."""
    utils.thick_line()
    start_time_test = time.time()

    test_params = dict(
        cfg=self.cfg,
        multi_gpu=self.multi_gpu,
        version=self.cfg.VERSION,
        is_training=is_training,
        epoch_train=epoch,
        step_train=step,
        clf_arch_info=self.clf_arch_info,
        rec_arch_info=self.rec_arch_info
    )

    if mode == 'single':
      print('Testing on Single-object test set...')
      tester_ = Test
    elif mode == 'multi_obj':
      print('Testing on Multi-object test set...')
      tester_ = TestMultiObjects
    elif mode == 'oracle':
      print('Testing on Oracles test set...')
      tester_ = TestOracle
    else:
      raise ValueError('Wrong mode name')

    tester_(**test_params).tester(
        sess, self.inputs, self.labels, self.input_imgs,
        self.preds, self.rec_images, start_time_test,
        self.loss, self.accuracy, self.clf_loss, self.rec_loss)

  def _trainer(self, sess):

    utils.thick_line()
    print('Training...')

    # Merge all the summaries and create writers
    train_summary_path = join(self.summary_path, 'train')
    valid_summary_path = join(self.summary_path, 'valid')
    utils.check_dir([train_summary_path, valid_summary_path])

    utils.thin_line()
    print('Generating TensorFLow summary writer...')
    train_writer = tf.summary.FileWriter(train_summary_path, sess.graph)
    valid_writer = tf.summary.FileWriter(valid_summary_path)

    sess.run(tf.global_variables_initializer())
    step = 0

    for epoch_i in range(self.cfg.EPOCHS):

      epoch_start_time = time.time()
      utils.thick_line()
      print('Training on epoch: {}/{}'.format(epoch_i + 1, self.cfg.EPOCHS))

      utils.thin_line()
      train_batch_generator = utils.get_batches(
          self.x_train, self.y_train, self.cfg.BATCH_SIZE, imgs=self.imgs_train)

      if self.cfg.DISPLAY_STEP:
        iterator = range(self.n_batch_train)
        silent = False
      else:
        iterator = tqdm(range(self.n_batch_train),
                        total=self.n_batch_train,
                        ncols=100, unit=' batch')
        silent = True

      for _ in iterator:

        step += 1
        x_batch, y_batch, imgs_batch = next(train_batch_generator)

        # Training optimizer
        sess.run(self.optimizer, feed_dict={self.inputs: x_batch,
                                            self.labels: y_batch,
                                            self.input_imgs: imgs_batch,
                                            self.step: step-1,
                                            self.is_training: True})

        # Display training information
        if self.cfg.DISPLAY_STEP:
          if step % self.cfg.DISPLAY_STEP == 0:
            self._display_status(sess, x_batch, y_batch,
                                 imgs_batch, epoch_i, step-1)

        # Save training logs
        if self.cfg.SAVE_LOG_STEP:
          if step % self.cfg.SAVE_LOG_STEP == 0:
            self._save_logs(sess, train_writer, valid_writer,
                            x_batch, y_batch, imgs_batch, epoch_i, step-1)

        # Save reconstruction images
        if self.cfg.SAVE_IMAGE_STEP:
          if self.cfg.WITH_REC:
            if step % self.cfg.SAVE_IMAGE_STEP == 0:
              self._save_images(
                  sess, self.train_image_path, x_batch, y_batch, imgs_batch,
                  step-1, epoch_i=epoch_i, silent=silent)

        # Save models
        if self.cfg.SAVE_MODEL_MODE == 'per_batch':
          if step % self.cfg.SAVE_MODEL_STEP == 0:
            self._save_model(sess, self.saver, step-1, silent=silent)

        # Evaluate on full set
        if self.cfg.FULL_SET_EVAL_MODE == 'per_batch':
          if step % self.cfg.FULL_SET_EVAL_STEP == 0:
            self._eval_on_full_set(sess, epoch_i, step-1, silent=silent)

      # Save model per epoch
      if self.cfg.SAVE_MODEL_MODE == 'per_epoch':
        if (epoch_i + 1) % self.cfg.SAVE_MODEL_STEP == 0:
          self._save_model(sess, self.saver, epoch_i)

      # Evaluate on valid set per epoch
      if self.cfg.FULL_SET_EVAL_MODE == 'per_epoch':
        if (epoch_i + 1) % self.cfg.FULL_SET_EVAL_STEP == 0:
          self._eval_on_full_set(sess, epoch_i, step-1)

      # Evaluate on test set per epoch
      if self.cfg.TEST_SO_MODE == 'per_epoch':
        self._test(sess, is_training=True,
                   epoch=epoch_i, step=step, mode='single')

      # Evaluate on multi-objects test set per epoch
      if self.cfg.TEST_MO_MODE == 'per_epoch':
        self._test(sess, is_training=True,
                   epoch=epoch_i, step=step, mode='multi_obj')

      # Evaluate on Oracles test set per epoch
      if self.cfg.DATABASE_NAME == 'radical':
        if self.cfg.TEST_ORACLE_MODE == 'per_epoch':
          self._test(sess, is_training=True,
                     epoch=epoch_i, step=step, mode='oracle')

      utils.thin_line()
      print('Epoch {}/{} done! Using time: {:.2f}'
            .format(epoch_i + 1, self.cfg.EPOCHS,
                    time.time() - epoch_start_time))

    utils.thick_line()
    print('Training finished! Using time: {:.2f}'
          .format(time.time() - self.start_time))
    utils.thick_line()

    # Evaluate on test set after training
    if self.cfg.TEST_SO_MODE == 'after_training':
      self._test(sess, is_training=True, epoch='end', mode='single')

    # Evaluate on multi-objects test set after training
    if self.cfg.TEST_MO_MODE == 'after_training':
      self._test(sess, is_training=True, epoch='end', mode='multi_obj')

    # Evaluate on Oracles test set after training
    if self.cfg.DATABASE_NAME == 'radical':
      if self.cfg.TEST_ORACLE_MODE == 'after_training':
        self._test(sess, is_training=True, epoch='end', mode='oracle')

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
  parser.add_argument('-g', '--gpu', nargs=1,
                      choices=[0, 1], type=int, metavar='',
                      help="Run single-gpu version."
                           "Choose the GPU from: {!s}".format([0, 1]))
  parser.add_argument('-bs', '--batch_size',
                      type=int, metavar='',
                      help="Set batch size.")
  parser.add_argument('-tn', '--task_number',
                      type=int, metavar='',
                      help="Set task number.")
  parser.add_argument('-m', '--mgpu', action="store_true",
                      help="Run multi-gpu version.")
  parser.add_argument('-t', '--mtask', action="store_true",
                      help="Run multi-tasks version.")
  parser.add_argument('-b', '--baseline', action="store_true",
                      help="Use baseline architecture and configurations.")
  args = parser.parse_args()

  if args.mtask:
    utils.thick_line()
    print('Running multi-tasks version.')
    mode_ = 'multi-tasks'
  elif args.mgpu:
    utils.thick_line()
    print('Running multi-gpu version.')
    mode_ = 'multi-gpu'
  elif args.gpu:
    utils.thick_line()
    print('Running single version. Using /gpu: %d' % args.gpu)
    environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    mode_ = 'single-gpu'
  else:
    utils.thick_line()
    print('Running normal version.')
    mode_ = 'normal'

  if args.baseline:
    print('Running baseline model.')
    utils.thick_line()
    arch_ = basel_arch
    config_ = basel_cfg
  else:
    arch_ = caps_arch
    config_ = config

  if args.batch_size:
    config_.BATCH_SIZE = args.batch_size

  if args.task_number:
    config_.TASK_NUMBER = args.task_number

  Main(config_, arch_, mode=mode_).train()
