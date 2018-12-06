from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from os.path import join, isdir
from sklearn.metrics import \
  precision_score, recall_score, f1_score, accuracy_score

from config import config
from models import utils


class Test(object):

  def __init__(self, cfg):

    # Config
    self.cfg = cfg

    # Get checkpoint path
    self.checkpoint_path = join(
        cfg.CHECKPOINT_PATH,
        '{}/models.ckpt-{}'.format(
            self.cfg.TEST_VERSION, self.cfg.TEST_CKP_IDX)
    )

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
      if self.cfg.TEST_SAVE_IMAGE_STEP:
        utils.check_dir([self.test_image_path])

    # Save config
    utils.save_config_log(self.test_log_path, self.cfg)

    # Load data
    self.x_test, self.y_test = self._load_data()

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
      accuracy_ = loaded_graph.get_tensor_by_name("accuracy:0")
      loss_ = loaded_graph.get_tensor_by_name("loss:0")

      if self.cfg.TEST_WITH_RECONSTRUCTION:
        clf_loss_ = loaded_graph.get_tensor_by_name("classifier_loss:0")
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

  def _eval(self,
            sess,
            inputs,
            labels,
            loss,
            accuracy,
            clf_loss,
            rec_loss,
            rec_images,
            x,
            y):

    if self.cfg.TEST_WITH_RECONSTRUCTION:
      loss_, clf_loss_, rec_loss_, acc_ = \
        sess.run([loss, clf_loss, rec_loss, accuracy],
                 feed_dict={inputs: x, labels: y})
      # Save reconstruct images
      if self.cfg.TEST_SAVE_IMAGE_STEP:
          self._save_images(sess, rec_images, inputs, labels, x, y)
    else:
      loss_, acc_ = \
        sess.run([loss, accuracy],
                 feed_dict={inputs: x, labels: y})
      clf_loss_, rec_loss_ = None, None

    return loss_, clf_loss_, rec_loss_, acc_

  def _eval_on_batches(self,
                       sess,
                       inputs,
                       labels,
                       loss,
                       accuracy,
                       clf_loss,
                       rec_loss,
                       rec_images,
                       x,
                       y):
    """Calculate losses and accuracies of full train set."""
    loss_all = []
    acc_all = []
    clf_loss_all = []
    rec_loss_all = []
    step = 0
    _batch_generator = utils.get_batches(x, y, self.cfg.TEST_BATCH_SIZE)
    n_batch = len(y) // self.cfg.TEST_BATCH_SIZE

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

      # Get losses and accuracies
      if self.cfg.TEST_BATCH_SIZE:
        loss_test, clf_loss_test, rec_loss_test, acc_test = \
            self._eval_on_batches(
                sess, inputs, labels, loss, accuracy,
                clf_loss, rec_loss, rec_images,
                self.x_test, self.y_test)
      else:
        loss_test, clf_loss_test, rec_loss_test, acc_test = \
          self._eval(sess, inputs, labels, loss, accuracy, clf_loss,
                     rec_loss, rec_images, self.x_test, self.y_test)

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


class TestMultiObjects(object):

  def __init__(self, cfg):

    # Config
    self.cfg = cfg

    # Get checkpoint path
    self.checkpoint_path = join(
        cfg.CHECKPOINT_PATH,
        '{}/models.ckpt-{}'.format(
            self.cfg.TEST_VERSION, self.cfg.TEST_CKP_IDX)
    )

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
      if self.cfg.TEST_SAVE_IMAGE_STEP:
        utils.check_dir([self.test_image_path])

    # Save config
    utils.save_config_log(self.test_log_path, self.cfg)

    # Load data
    self.x_test, self.y_test = self._load_data()

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

      inputs_ = loaded_graph.get_tensor_by_name("inputs:0")
      labels_ = loaded_graph.get_tensor_by_name("labels:0")
      preds_ = loaded_graph.get_tensor_by_name("preds:0")
      loss_ = loaded_graph.get_tensor_by_name("loss:0")

      if self.cfg.TEST_WITH_RECONSTRUCTION:
        clf_loss_ = loaded_graph.get_tensor_by_name("classifier_loss:0")
        rec_loss_ = loaded_graph.get_tensor_by_name("rec_loss:0")
        rec_images_ = loaded_graph.get_tensor_by_name("rec_images:0")
        return inputs_, labels_, preds_, loss_, \
            clf_loss_, rec_loss_, rec_images_
      else:
        return inputs_, labels_, preds_, loss_

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

  def _eval(self,
            sess,
            inputs,
            labels,
            preds,
            loss,
            clf_loss,
            rec_loss,
            rec_images,
            x,
            y):
    """Evaluation."""
    if self.cfg.TEST_WITH_RECONSTRUCTION:
      loss_, clf_loss_, rec_loss_, preds_ = \
        sess.run([loss, clf_loss, rec_loss, preds],
                 feed_dict={inputs: x, labels: y})
      # Save reconstruct images
      if self.cfg.TEST_SAVE_IMAGE_STEP:
        self._save_images(sess, rec_images, inputs, labels, x, y)
    else:
      loss_, preds_ = \
        sess.run([loss, preds],
                 feed_dict={inputs: x, labels: y})
      clf_loss_, rec_loss_ = None, None

    return loss_, clf_loss_, rec_loss_, preds_

  def _eval_on_batches(self,
                       sess,
                       inputs,
                       labels,
                       preds,
                       loss,
                       clf_loss,
                       rec_loss,
                       rec_images,
                       x,
                       y):
    """Calculate losses and accuracies of full train set."""
    loss_all = []
    clf_loss_all = []
    rec_loss_all = []
    pred_all = []
    step = 0
    _batch_generator = utils.get_batches_all(x, y, self.cfg.TEST_BATCH_SIZE)
    n_batch = (len(y) // self.cfg.TEST_BATCH_SIZE) + 1

    for _ in tqdm(range(n_batch), total=n_batch,
                  ncols=100, unit=' batches'):
      step += 1
      x_batch, y_batch = next(_batch_generator)

      # The last batch which has less examples
      len_batch = len(x_batch)
      if len_batch != self.cfg.TEST_BATCH_SIZE:
        for i in range(self.cfg.TEST_BATCH_SIZE - len_batch):
          x_batch = np.append(x_batch, np.expand_dims(
              np.zeros_like(x_batch[0]), axis=0), axis=0)
        assert len(x_batch) == self.cfg.TEST_BATCH_SIZE

      if self.cfg.TEST_WITH_RECONSTRUCTION:
        loss_i, clf_loss_i, rec_loss_i, pred_i = \
            sess.run([loss, clf_loss, rec_loss, preds],
                     feed_dict={inputs: x_batch, labels: y_batch})

        if len_batch != self.cfg.TEST_BATCH_SIZE:
          pred_i = pred_i[:len_batch]
        else:
          loss_all.append(loss_i)
          clf_loss_all.append(clf_loss_i)
          rec_loss_all.append(rec_loss_i)
      else:
        loss_i, pred_i = \
          sess.run([loss, preds],
                   feed_dict={inputs: x_batch, labels: y_batch})
        if len_batch != self.cfg.TEST_BATCH_SIZE:
          pred_i = pred_i[:len_batch]
        else:
          loss_all.append(loss_i)

      pred_all.extend(pred_i.to_list())

      # Save reconstruct images
      if self.cfg.TEST_SAVE_IMAGE_STEP:
        if step % self.cfg.TEST_SAVE_IMAGE_STEP == 0:
          self._save_images(sess, rec_images, inputs, labels,
                            x_batch, y_batch, step)

    if self.cfg.TEST_WITH_RECONSTRUCTION:
      clf_loss_ = sum(clf_loss_all) / len(clf_loss_all)
      rec_loss_ = sum(rec_loss_all) / len(rec_loss_all)
    else:
      clf_loss_, rec_loss_ = None, None
    loss_ = sum(loss_all) / len(loss_all)

    assert len(pred_all) == len(x)
    preds_ = np.array(pred_all)

    return loss_, clf_loss_, rec_loss_, preds_

  def _get_preds(self,
                 preds_vec,
                 predict_mode='top_n',
                 predict_num=2
                 ):
    """Get predictions

     -> [0, 0, 1, ..., 0, 1, 0] as labels
     """
    preds = np.array(preds_vec)
    if predict_mode == 'top_n':
      for pred_i in preds:
        pos_idx = np.argsort(preds)[-predict_num:]
        neg_idx = np.argsort(preds)[:-predict_num]
        pred_i[pos_idx] = 1
        pred_i[neg_idx] = 0
    elif predict_mode == 'length_rate':
      for pred_i in preds:
        max_ = pred_i.max()
        pred_i[pred_i < (max_ * predict_num)] = 0
        pred_i[pred_i >= (max_ * predict_num)] = 1
    else:
      raise ValueError('Wrong Mode Name! Find {}!'.format(predict_mode))

    if self.cfg.SAVE_TEST_PRED:
      utils.save_test_pred(self.test_log_path, self.y_test, preds, preds_vec)

    return preds

  def _get_multi_obj_scores(self,
                            preds_vec,
                            predict_mode='top_n',
                            predict_num=2):
    """Get accuracy for multi-objects detection."""
    preds = self._get_preds(
        preds_vec=preds_vec,
        predict_mode=predict_mode,
        predict_num=predict_num)

    # Calculate scores manually
    precision_manual = []
    recall_manual = []
    accuracy_manual = []
    f1score_manual = []
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
      precision_manual.append(precision_)
      recall_ = tp / (tp + fn)
      recall_manual.append(recall_)
      accuracy_manual.append((tp + tn) / (tp + fp + tn + fn))
      f1score_manual.append(precision_ * recall_ / 2 * (precision_ + recall_))
    precision_manual = np.mean(precision_manual)
    recall_manual = np.mean(recall_manual)
    accuracy_manual = np.mean(accuracy_manual)
    f1score_manual = np.mean(f1score_manual)

    # Calculate scores by using scikit-learn tools
    precision = precision_score(self.y_test, preds, average='binary')
    recall = recall_score(self.y_test, preds, average='binary')
    accuracy = accuracy_score(self.y_test, preds)
    f1score = f1_score(self.y_test, preds, average='binary')

    # Print evaluation information
    utils.print_multi_obj_eval(
        precision_manual, recall_manual,
        accuracy_manual, f1score_manual,
        precision, recall, accuracy, f1score)

    return precision, recall, accuracy, f1score

  def test(self):
    """Test models."""
    start_time = time.time()
    tf.reset_default_graph()
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:

      # Load saved models
      loader = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
      loader.restore(sess, self.checkpoint_path)

      # Get Tensors from loaded models
      if self.cfg.TEST_WITH_RECONSTRUCTION:
        inputs, labels, preds, loss, \
            clf_loss, rec_loss, rec_images = \
            self._get_tensors(loaded_graph)
      else:
        inputs, labels, preds, loss = self._get_tensors(loaded_graph)
        clf_loss, rec_loss, rec_images = None, None, None

      utils.thick_line()
      print('Testing on test set...')

      utils.thin_line()
      print('Calculating loss and accuracy of test set...')

      # Get losses and accuracies
      if self.cfg.TEST_BATCH_SIZE:
        loss_test, clf_loss_test, rec_loss_test, preds_vec_test = \
            self._eval_on_batches(
                sess, inputs, labels, preds, loss,
                clf_loss, rec_loss, rec_images,
                self.x_test, self.y_test)
      else:
        loss_test, clf_loss_test, rec_loss_test, preds_vec_test = \
          self._eval(sess, inputs, labels, preds, loss, clf_loss,
                     rec_loss, rec_images, self.x_test, self.y_test)

      precision, recall, accuracy, f1score = \
          self._get_multi_obj_scores(preds_vec_test,
                                     predict_mode='top_n',
                                     predict_num=2)

      utils.save_multi_obj_scores(
          self.test_log_path, loss_test, clf_loss_test,
          rec_loss_test, self.cfg.TEST_WITH_RECONSTRUCTION,
          precision, recall, accuracy, f1score)

      # Print losses and accuracy
      utils.thin_line()
      print('Test_Loss: {:.4f}'.format(loss_test))
      if self.cfg.TEST_WITH_RECONSTRUCTION:
        print('Test_Classifier_Loss: {:.4f}\n'.format(clf_loss_test),
              'Test_Reconstruction_Loss: {:.4f}'.format(rec_loss_test))

      utils.thin_line()
      print('Testing finished! Using time: {:.2f}'
            .format(time.time() - start_time))
      utils.thick_line()


if __name__ == '__main__':

  Test_ = Test(config)
  Test_.test()
