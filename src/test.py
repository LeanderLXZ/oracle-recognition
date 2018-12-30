from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import re
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
from os import listdir
from os.path import join, isdir
# from sklearn.metrics import \
#   precision_score, recall_score, f1_score, accuracy_score

from config import config
from baseline_config import config as basel_cfg
from models import utils


class Test(object):

  def __init__(self,
               cfg,
               multi_gpu=False,
               version=None,
               load_last_ckp=True):

    # Config
    self.cfg = cfg
    self.multi_gpu = multi_gpu
    self.version = version

    # Checkpoint index
    if load_last_ckp:
      ckp_indices = []
      for f_name in listdir(join(self.cfg.CHECKPOINT_PATH, version)):
        m = re.match('.*-(\d*).meta', f_name)
        if m:
          ckp_indices.append(int(m.group(1)))
      self.ckp_idx = max(ckp_indices)
    else:
      self.ckp_idx = self.cfg.TEST_CKP_IDX

    # Get paths for testing
    self.checkpoint_path, self.test_log_path, self.test_image_path = \
        self._get_paths()

    # Save config
    utils.save_config_log(self.test_log_path, self.cfg)

    # Load data
    self.x_test, self.y_test = self._load_data()

  def _get_paths(self):
    """Get paths for testing."""
    # Get checkpoint path
    checkpoint_path = join(
        self.cfg.CHECKPOINT_PATH,
        '{}/models.ckpt-{}'.format(self.version, self.ckp_idx))

    # Get log path, append information if the directory exist.
    test_log_path_ = join(
        self.cfg.TEST_LOG_PATH,
        '{}-{}'.format(self.version, self.ckp_idx))
    test_log_path = test_log_path_
    i_append_info = 0
    while isdir(test_log_path):
      i_append_info += 1
      test_log_path = test_log_path_ + '({})'.format(i_append_info)

    # Path for saving images
    test_image_path = join(test_log_path, 'images')

    # Check directory of paths
    utils.check_dir([test_log_path])
    if self.cfg.TEST_WITH_REC:
      if self.cfg.TEST_SAVE_IMAGE_STEP:
        utils.check_dir([test_image_path])

    return checkpoint_path, test_log_path, test_image_path

  def _load_data(self):
    utils.thick_line()
    print('Loading data...')
    utils.thin_line()
    preprocessed_path_ = join(self.cfg.DPP_DATA_PATH, self.cfg.DATABASE_NAME)
    x = utils.load_data_from_pkl(
        join(preprocessed_path_, 'x_test.p'))
    y = utils.load_data_from_pkl(
        join(preprocessed_path_, 'y_test.p'))
    return x, y

  def _get_tensors(self, loaded_graph):
    """Get inputs, labels, loss, and accuracy tensor from <loaded_graph>."""
    with loaded_graph.as_default():

      utils.thin_line()
      print('Loading graph and tensors...')

      inputs_ = loaded_graph.get_tensor_by_name("inputs:0")
      labels_ = loaded_graph.get_tensor_by_name("labels:0")

      if self.multi_gpu:
        accuracy_ = loaded_graph.get_tensor_by_name("total_acc:0")
        loss_ = loaded_graph.get_tensor_by_name("total_loss:0")
        if self.cfg.TEST_WITH_REC:
          clf_loss_ = loaded_graph.get_tensor_by_name("total_clf_loss:0")
          rec_loss_ = loaded_graph.get_tensor_by_name("total_rec_loss:0")
          rec_images_ = loaded_graph.get_tensor_by_name("total_rec_images:0")
          return inputs_, labels_, loss_, accuracy_, \
              clf_loss_, rec_loss_, rec_images_
        else:
          return inputs_, labels_, loss_, accuracy_
      else:
        accuracy_ = loaded_graph.get_tensor_by_name("accuracy:0")
        loss_ = loaded_graph.get_tensor_by_name("loss:0")
        if self.cfg.TEST_WITH_REC:
          clf_loss_ = loaded_graph.get_tensor_by_name("clf_loss:0")
          rec_loss_ = loaded_graph.get_tensor_by_name("rec_loss:0")
          rec_images_ = loaded_graph.get_tensor_by_name("rec_images:0")
          return inputs_, labels_, loss_, accuracy_, \
              clf_loss_, rec_loss_, rec_images_
        else:
          return inputs_, labels_, loss_, accuracy_

  def _save_images(self,
                   sess,
                   rec_images,
                   inputs,
                   labels,
                   x,
                   y,
                   step=None):
    """Save reconstructed images."""
    rec_images_ = sess.run(
        rec_images, feed_dict={inputs: x, labels: y})

    utils.save_imgs(
        real_imgs=x,
        rec_imgs=rec_images_,
        img_path=self.test_image_path,
        database_name=self.cfg.DATABASE_NAME,
        max_img_in_col=self.cfg.MAX_IMAGE_IN_COL,
        step=step,
        silent=True,
        test_flag=True)

  def _eval_on_batches(self,
                       sess,
                       inputs,
                       labels,
                       loss,
                       accuracy,
                       clf_loss,
                       rec_loss,
                       rec_images):
    """Calculate losses and accuracies of full train set."""
    loss_all = []
    acc_all = []
    clf_loss_all = []
    rec_loss_all = []
    step = 0
    _batch_generator = utils.get_batches(
        self.x_test, self.y_test, self.cfg.TEST_BATCH_SIZE)
    n_batch = len(self.y_test) // self.cfg.TEST_BATCH_SIZE

    if self.cfg.TEST_WITH_REC:
      for _ in tqdm(range(n_batch), total=n_batch,
                    ncols=100, unit=' batch'):
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
        if self.cfg.TEST_SAVE_IMAGE_STEP:
          if step % self.cfg.TEST_SAVE_IMAGE_STEP == 0:
            self._save_images(sess, rec_images, inputs, labels,
                              x_batch, y_batch, step)

      clf_loss_ = sum(clf_loss_all) / len(clf_loss_all)
      rec_loss_ = sum(rec_loss_all) / len(rec_loss_all)

    else:
      for _ in tqdm(range(n_batch), total=n_batch,
                    ncols=100, unit=' batches'):
        x_batch, y_batch = next(_batch_generator)
        loss_i, acc_i = \
            sess.run([loss, accuracy],
                     feed_dict={inputs: x_batch, labels: y_batch})
        loss_all.append(loss_i)
        acc_all.append(acc_i)
      clf_loss_, rec_loss_ = None, None

    loss_ = sum(loss_all) / len(loss_all)
    acc_ = sum(acc_all) / len(acc_all)

    return loss_, clf_loss_, rec_loss_, acc_

  def test(self):
    """Test models."""
    start_time = time.time()
    tf.reset_default_graph()
    loaded_graph = tf.Graph()

    utils.thick_line()
    print('Testing on test set...')

    with tf.Session(graph=loaded_graph) as sess:

      # Load saved models
      loader = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
      loader.restore(sess, self.checkpoint_path)

      # Get Tensors from loaded models
      if self.cfg.TEST_WITH_REC:
        inputs, labels, loss, accuracy, \
            clf_loss, rec_loss, rec_images = \
            self._get_tensors(loaded_graph)
      else:
        inputs, labels, loss, accuracy = self._get_tensors(loaded_graph)
        clf_loss, rec_loss, rec_images = None, None, None

      utils.thin_line()
      print('Calculating loss and accuracy of test set...')

      # Get losses and accuracies
      loss_test, clf_loss_test, rec_loss_test, acc_test = \
          self._eval_on_batches(
              sess, inputs, labels, loss, accuracy,
              clf_loss, rec_loss, rec_images)

      # Print losses and accuracy
      utils.thin_line()
      print('Test_Loss: {:.4f}'.format(loss_test))
      if self.cfg.TEST_WITH_REC:
        print('Test_clf_loss: {:.4f}\n'.format(clf_loss_test),
              'Test_REC_LOSS: {:.4f}'.format(rec_loss_test))
      print('Test_Accuracy: {:.2f}%'.format(acc_test * 100))

      # Save test log
      utils.save_test_log(
          self.test_log_path, loss_test, acc_test, clf_loss_test,
          rec_loss_test, self.cfg.TEST_WITH_REC)

      utils.thin_line()
      print('Testing finished! Using time: {:.2f}'
            .format(time.time() - start_time))
      utils.thick_line()


class TestMultiObjects(object):

  def __init__(self,
               cfg,
               multi_gpu=False,
               version=None,
               load_last_ckp=True):

    # Config
    self.cfg = cfg
    self.multi_gpu = multi_gpu
    self.version = version

    # Checkpoint index
    if load_last_ckp:
      ckp_indices = []
      for f_name in listdir(join(self.cfg.CHECKPOINT_PATH, version)):
        m = re.match('.*-(\d*).meta', f_name)
        if m:
          ckp_indices.append(int(m.group(1)))
      self.ckp_idx = max(ckp_indices)
    else:
      self.ckp_idx = self.cfg.TEST_CKP_IDX

    # Get paths for testing
    self.checkpoint_path, self.test_log_path, self.test_image_path = \
        self._get_paths()

    # Save config
    utils.save_config_log(self.test_log_path, self.cfg)

    # Load data
    self.x_test, self.y_test = self._load_data()

  def _get_paths(self):
    """Get paths for testing."""
    # Get checkpoint path
    checkpoint_path = join(
        self.cfg.CHECKPOINT_PATH,
        '{}/models.ckpt-{}'.format(self.version, self.ckp_idx))

    # Get log path, append information if the directory exist.
    test_log_path_ = join(
        self.cfg.TEST_LOG_PATH,
        '{}-{}'.format(self.version, self.ckp_idx))
    test_log_path = test_log_path_ + '_multi_obj'
    i_append_info = 0
    while isdir(test_log_path):
      i_append_info += 1
      test_log_path = test_log_path_ + '({})'.format(i_append_info)

    # Path for saving images
    test_image_path = join(test_log_path, 'images')

    # Check directory of paths
    utils.check_dir([test_log_path])
    if self.cfg.TEST_WITH_REC:
      if self.cfg.TEST_SAVE_IMAGE_STEP:
        utils.check_dir([test_image_path])

    return checkpoint_path, test_log_path, test_image_path

  def _load_data(self):
    utils.thick_line()
    print('Loading data...')
    utils.thin_line()
    preprocessed_path_ = join(self.cfg.DPP_DATA_PATH, self.cfg.DATABASE_NAME)
    x = utils.load_data_from_pkl(
        join(preprocessed_path_, 'x_test_mul.p'))
    y = utils.load_data_from_pkl(
        join(preprocessed_path_, 'y_test_mul.p'))
    return x, y

  def _get_tensors(self, loaded_graph):
    """Get inputs, labels, loss, and accuracy tensor from <loaded_graph>."""
    with loaded_graph.as_default():

      utils.thin_line()
      print('Loading graph and tensors...')

      inputs_ = loaded_graph.get_tensor_by_name('inputs:0')
      labels_ = loaded_graph.get_tensor_by_name('labels:0')

      if self.multi_gpu:
        preds_ = loaded_graph.get_tensor_by_name('total_preds:0')
        if self.cfg.TEST_WITH_REC:
          rec_images_ = loaded_graph.get_tensor_by_name('total_rec_images:0')
          return inputs_, labels_, preds_, rec_images_
        else:
          return inputs_, labels_, preds_
      else:
        preds_ = loaded_graph.get_tensor_by_name('preds:0')
        if self.cfg.TEST_WITH_REC:
          rec_images_ = loaded_graph.get_tensor_by_name('rec_images:0')
          return inputs_, labels_, preds_, rec_images_
        else:
          return inputs_, labels_, preds_

  def _get_preds_vector(self,
                        sess,
                        inputs,
                        preds):
    """Get prediction vectors of full train set."""
    utils.thin_line()
    print('Getting prediction vectors...')
    pred_all = []
    _batch_generator = utils.get_batches_all(
        self.x_test, self.cfg.TEST_BATCH_SIZE)
    n_batch = (len(self.x_test) // self.cfg.TEST_BATCH_SIZE) + 1

    for _ in tqdm(range(n_batch), total=n_batch,
                  ncols=100, unit=' batch'):
      x_batch = next(_batch_generator)

      # The last batch which has less examples
      len_batch = len(x_batch)
      if len_batch != self.cfg.TEST_BATCH_SIZE:
        for i in range(self.cfg.TEST_BATCH_SIZE - len_batch):
          x_batch = np.append(x_batch, np.expand_dims(
              np.zeros_like(x_batch[0]), axis=0), axis=0)
        assert len(x_batch) == self.cfg.TEST_BATCH_SIZE

      pred_i = sess.run(preds, feed_dict={inputs: x_batch})
      if len_batch != self.cfg.TEST_BATCH_SIZE:
        pred_i = pred_i[:len_batch]
      pred_all.extend(list(pred_i))

    assert len(pred_all) == len(self.x_test), (len(pred_all), len(self.x_test))
    return np.array(pred_all)

  def _get_preds_binary(self, preds_vec):
    """Get binary predictions.

     -> [0, 0, 1, ..., 0, 1, 0] as labels
     """
    utils.thin_line()
    print('Converting prediction vectors to binaries...')
    preds = np.array(preds_vec)
    if self.cfg.MOD_PRED_MODE == 'top_n':
      for pred_i in preds:
        pos_idx = np.argsort(pred_i)[-self.cfg.MOD_PRED_MAX_NUM:]
        neg_idx = np.argsort(pred_i)[:-self.cfg.MOD_PRED_MAX_NUM]
        pred_i[pos_idx] = 1
        pred_i[neg_idx] = 0
    elif self.cfg.MOD_PRED_MODE == 'length_rate':
      for pred_i in preds:
        pred_i_copy = pred_i.copy()
        max_ = pred_i.max()
        pred_i[pred_i < (max_ * self.cfg.MOD_PRED_THRESHOLD)] = 0
        pred_i[pred_i >= (max_ * self.cfg.MOD_PRED_THRESHOLD)] = 1
        if np.sum(pred_i) > self.cfg.MOD_PRED_MAX_NUM:
          pos_idx = np.argsort(pred_i_copy)[-self.cfg.MOD_PRED_MAX_NUM:]
          neg_idx = np.argsort(pred_i_copy)[:-self.cfg.MOD_PRED_MAX_NUM]
          pred_i[pos_idx] = 1
          pred_i[neg_idx] = 0
    else:
      raise ValueError(
          'Wrong Mode Name! Find {}!'.format(self.cfg.MOD_PRED_MODE))

    if self.cfg.SAVE_TEST_PRED:
      utils.save_test_pred(self.test_log_path, self.y_test, preds, preds_vec)

    return np.array(preds, dtype=int)

  def _get_multi_obj_scores(self, preds):
    """Get evaluation scores for multi-objects detection."""
    utils.thin_line()
    print('Calculating evaluation scores for multi-objects detection...')

    def _f_beta_score(p, r, beta):
      if p + r == 0:
        return 0.
      else:
        return ((1 + (beta ** 2)) * p * r) / ((beta ** 2) * p + r)

    # Calculate scores manually
    precision = []
    recall = []
    accuracy = []
    f1score = []
    f05score = []
    f2score = []
    for y_pred, y_true in zip(preds, self.y_test):
      # true positive
      tp = np.sum(np.multiply(y_true, y_pred))
      # false positive
      fp = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
      # false negative
      fn = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
      # true negative
      tn = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
      precision_ = tp / (tp + fp)
      accuracy_ = (tp + tn) / (tp + fp + tn + fn)
      recall_ = tp / (tp + fn)
      precision.append(precision_)
      accuracy.append(accuracy_)
      recall.append(recall_)
      f1score.append(_f_beta_score(precision_, recall_, 1.))
      f05score.append(_f_beta_score(precision_, recall_, 0.5))
      f2score.append(_f_beta_score(precision_, recall_, 2.))
    precision = np.mean(precision)
    recall = np.mean(recall)
    accuracy = np.mean(accuracy)
    f1score = np.mean(f1score)
    f05score = np.mean(f05score)
    f2score = np.mean(f2score)

    # true positive
    tp = np.sum(np.multiply(preds, self.y_test))
    print('TRUE POSITIVE: ', tp)
    # false positive
    fp = np.sum(np.logical_and(np.equal(self.y_test, 0), np.equal(preds, 1)))
    print('FALSE POSITIVE: ', fp)
    # false negative
    fn = np.sum(np.logical_and(np.equal(self.y_test, 1), np.equal(preds, 0)))
    print('TRUE NEGATIVE: ', fn)
    # true negative
    tn = np.sum(np.logical_and(np.equal(self.y_test, 0), np.equal(preds, 0)))
    print('FALSE NEGATIVE: ', tn)

    # Calculate scores by using scikit-learn tools
    # precision = precision_score(self.y_test, preds, average='samples')
    # recall = recall_score(self.y_test, preds, average='samples')
    # accuracy = accuracy_score(self.y_test, preds)
    # f1score = f1_score(self.y_test, preds, average='samples')

    # Print evaluation information
    utils.print_multi_obj_eval(
        precision, recall, accuracy, f1score, f05score, f2score)

    # Save evaluation scores of multi-objects detection.
    utils.save_multi_obj_scores(
        self.test_log_path, precision, recall,
        accuracy, f1score, f05score, f2score)

  def _save_images(self,
                   sess,
                   rec_images,
                   inputs,
                   labels,
                   preds_binary,
                   preds_vector):
    """Save reconstructed images."""
    utils.thin_line()
    print('Getting reconstruction images...')
    if len(self.y_test) > self.cfg.MAX_IMAGE_IN_COL ** 2:
      n_test_img = self.cfg.MAX_IMAGE_IN_COL ** 2
      test_img_idx = np.random.choice(len(self.y_test), n_test_img)
    else:
      test_img_idx = list(range(len(self.y_test)))

    rec_images_ = []
    preds_vec_ = []

    if self.cfg.LABEL_FOR_TEST == 'pred':
      label_for_img = preds_binary
    elif self.cfg.LABEL_FOR_TEST == 'real':
      label_for_img = self.y_test
    else:
      raise ValueError('Wrong LABEL_FOR_TEST Name!')

    for x, y_hat, pred_ in tqdm(zip(self.x_test[test_img_idx],
                                    label_for_img[test_img_idx],
                                    preds_vector[test_img_idx]),
                                total=len(test_img_idx),
                                ncols=100, unit=' image'):
      # Get new x and y_hat list in which each y contain single object
      # [0, 1, 0, 1, 0] -> [[0, 1, 0, 0, 0],
      #                     [0, 0, 0, 1, 0]]
      x_new = []
      y_hat_new = []
      preds_vec_new = []
      for i, y_i in enumerate(y_hat):
        if y_i == 1:
          y_hat_new_i = np.zeros_like(y_hat)
          y_hat_new_i[i] = 1
          assert y_hat_new_i[i] == y_hat[i]
          x_new.append(x)
          y_hat_new.append(y_hat_new_i)
          preds_vec_new.append(pred_[i])
      preds_vec_.append(preds_vec_new)

      # Filling x and y tensor to batch size for testing
      # [[0, 1, 0, 0, 0],
      #  [0, 0, 0, 1, 0]] -> [[0, 1, 0, 0, 0],
      #                       [0, 0, 0, 1, 0],
      #                             ...
      #                       [0, 0, 0, 0, 0]]
      n_y = len(y_hat_new)
      assert n_y == int(np.sum(y_hat))
      if n_y > self.cfg.TEST_BATCH_SIZE:
        raise ValueError(
            'TEST_BATCH_SIZE Must Not Less Than {}!'.format(n_y))
      if n_y < self.cfg.TEST_BATCH_SIZE:
        for i in range(self.cfg.TEST_BATCH_SIZE - n_y):
          x_new = np.append(x_new, np.expand_dims(
              np.zeros_like(x), axis=0), axis=0)
          y_hat_new.append(np.zeros_like(y_hat))
      assert len(x_new) == self.cfg.TEST_BATCH_SIZE
      assert len(y_hat_new) == self.cfg.TEST_BATCH_SIZE

      # Get remake images which contain different objects
      # y_rec_imgs_ shape: [128, 28, 28, 1] for mnist
      y_rec_imgs_ = sess.run(
          rec_images, feed_dict={inputs: x_new, labels: y_hat_new})
      rec_images_.append(y_rec_imgs_[:n_y])

    # Get colorful overlapped images
    real_imgs_ = utils.img_black_to_color(
        self.x_test[test_img_idx], same=True)
    rec_imgs_overlap = []
    rec_imgs_no_overlap = []
    for idx, imgs in enumerate(rec_images_):
      imgs_colored = utils.img_black_to_color(imgs)
      imgs_overlap = utils.img_add_overlap(
          imgs=imgs_colored,
          merge=True,
          vec=preds_vec_[idx],
          # vec=None,
          gamma=0)
      imgs_no_overlap = utils.img_add_no_overlap(
          imgs=imgs_colored,
          num_mul_obj=self.cfg.NUM_MULTI_OBJECT,
          vec=preds_vec_[idx],
          img_mode='RGB',
          resize_filter=Image.ANTIALIAS)
      rec_imgs_overlap.append(imgs_overlap)
      rec_imgs_no_overlap.append(imgs_no_overlap)
    rec_imgs_overlap = np.array(rec_imgs_overlap)
    rec_imgs_no_overlap = np.array(rec_imgs_no_overlap)

    # Save images
    utils.save_imgs(
        real_imgs=real_imgs_,
        rec_imgs=rec_imgs_overlap,
        img_path=self.test_image_path,
        database_name=self.cfg.DATABASE_NAME,
        max_img_in_col=self.cfg.MAX_IMAGE_IN_COL,
        silent=False,
        test_flag=True,
        colorful=True,
        append_info='_overlap'
    )
    utils.save_imgs(
        real_imgs=real_imgs_,
        rec_imgs=rec_imgs_no_overlap,
        img_path=self.test_image_path,
        database_name=self.cfg.DATABASE_NAME,
        max_img_in_col=self.cfg.MAX_IMAGE_IN_COL,
        silent=False,
        test_flag=True,
        colorful=True,
        append_info='_no_overlap'
    )

  def test(self):
    """Test models."""
    start_time = time.time()
    tf.reset_default_graph()
    loaded_graph = tf.Graph()

    utils.thick_line()
    print('Testing on test set...')

    with tf.Session(graph=loaded_graph) as sess:

      # Load saved models
      loader = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
      loader.restore(sess, self.checkpoint_path)

      # Get Tensors from loaded models
      if self.cfg.TEST_WITH_REC:
        inputs, labels, preds, rec_images = self._get_tensors(loaded_graph)
      else:
        inputs, labels, preds = self._get_tensors(loaded_graph)
        rec_images = None

      utils.thin_line()
      print('Calculating loss and accuracy of test set...')

      # Get losses and accuracies
      preds_vec_test = self._get_preds_vector(sess, inputs, preds)

      # Get binary predictions
      preds_binary = self._get_preds_binary(preds_vec=preds_vec_test)

      # Get evaluation scores for multi-objects detection.
      self._get_multi_obj_scores(preds_binary)

      # Save reconstruction images of multi-objects detection
      self._save_images(sess, rec_images, inputs,
                        labels, preds_binary, preds_vec_test)

      utils.thin_line()
      print('Testing finished! Using time: {:.2f}'
            .format(time.time() - start_time))
      utils.thick_line()


if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      description="Testing the model."
  )
  parser.add_argument('-b', '--baseline', action="store_true",
                      help="Use baseline configurations.")
  parser.add_argument('-mo', '--multi_obj', action="store_true",
                      help="Test multi objects detection.")
  parser.add_argument('-m', '--mgpu', action="store_true",
                      help="Test multi-gpu version.")
  args = parser.parse_args()

  if args.multi_obj:
    utils.thick_line()
    print('Testing multi objects detection.')
    utils.thick_line()
    Test_ = TestMultiObjects
  else:
    Test_ = Test

  if args.mgpu:
    utils.thick_line()
    print('Testing multi-gpu version.')
    multi_gpu_ = True
  else:
    multi_gpu_ = False

  if args.baseline:
    utils.thick_line()
    print('Running baseline model.')
    config_ = basel_cfg
  else:
    config_ = config

  load_last_ckp_ = False if config_.TEST_CKP_IDX else True

  Test_(cfg=config_,
        multi_gpu=multi_gpu_,
        version=config_.TEST_VERSION,
        load_last_ckp=load_last_ckp_).test()
