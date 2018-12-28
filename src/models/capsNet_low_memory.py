from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.capsNet_distribute import CapsNetDistribute


class CapsNetLowMemory(CapsNetDistribute):

  def __init__(self, cfg, model_arch):
    super(CapsNetLowMemory, self).__init__(cfg, model_arch)
    self.clf_arch_info = None
    self.rec_arch_info = None
    self.batch_size = cfg.BATCH_SIZE // cfg.GPU_NUMBER // cfg.TASK_NUMBER

  def _average_metrics_tower(self, loss_tower, acc_tower, preds_tower,
                             clf_loss_tower, rec_loss_tower, rec_images_tower):
    """Calculate average of metrics of a tower.

    Args:
      loss_tower: final losses of each task, list
      acc_tower: accuracies of each task, list
      preds_tower: predictions of each task, list
      clf_loss_tower: classifier losses of each task, list
      rec_loss_tower: reconstruction losses of each task, list
      rec_images_tower: reconstructed images of each task, list of 4D tensor

    Returns:
      tuple of metrics
    """
    n_task = float(len(loss_tower))

    loss_tower = tf.divide(
        tf.add_n(loss_tower), n_task, name='loss_tower')
    assert loss_tower.get_shape() == ()

    acc_tower = tf.divide(
        tf.add_n(acc_tower), n_task, name='acc_tower')
    assert acc_tower.get_shape() == ()

    preds_tower = tf.concat(preds_tower, axis=0, name='preds_tower')
    assert preds_tower.get_shape()[0] == self.cfg.BATCH_SIZE

    if self.cfg.WITH_REC:
      clf_loss_tower = tf.divide(
          tf.add_n(clf_loss_tower), n_task, name='clf_loss_tower')
      assert clf_loss_tower.get_shape() == ()

      rec_loss_tower = tf.divide(
          tf.add_n(rec_loss_tower), n_task, name='rec_loss_tower')
      assert rec_loss_tower.get_shape() == ()

      rec_images_tower = tf.concat(
          rec_images_tower, axis=0, name='rec_images_tower')
      assert rec_images_tower.get_shape() == (
        self.cfg.BATCH_SIZE // self.cfg.GPU_NUMBER,
        *rec_images_tower[0].get_shape().as_list()[1:])
    else:
      clf_loss_tower, rec_loss_tower, rec_images_tower = None, None, None

    return loss_tower, acc_tower, preds_tower, \
        clf_loss_tower, rec_loss_tower, rec_images_tower

  def _calc_on_gpu(self, gpu_idx, x_splits_tower, y_splits_tower,
                   image_size, is_training, optimizer):

    # Dequeues one batch for the GPU
    x_tower, y_tower = x_splits_tower[gpu_idx], y_splits_tower[gpu_idx]

    # Split data for each tower
    x_splits_task = tf.split(
        axis=0, num_or_size_splits=self.cfg.TASK_NUMBER, value=x_tower)
    y_splits_task = tf.split(
        axis=0, num_or_size_splits=self.cfg.TASK_NUMBER, value=y_tower)

    loss_tower, acc_tower, preds_tower, clf_loss_tower, \
        rec_loss_tower, rec_images_tower, grads_tower = \
        [], [], [], [], [], [], []
    for i in range(self.cfg.TASK_NUMBER):
      with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
        with tf.name_scope('task_%d' % i):

          # Dequeues one task
          x_task, y_task = x_splits_task[gpu_idx], y_splits_task[gpu_idx]

          # Calculate the loss for one tower.
          loss_task, acc_task, preds_task, clf_loss_task, \
              rec_loss_task, rec_images_task = \
              self._get_loss(x_task, y_task,
                             image_size, is_training=is_training)

          # Calculate the gradients on this tower.
          grads_task = optimizer.compute_gradients(loss_task)

          # Keep track of the gradients across all towers.
          grads_tower.append(grads_task)

          # Collect metrics of each tower
          loss_tower.append(loss_task)
          acc_tower.append(acc_task)
          clf_loss_tower.append(clf_loss_task)
          rec_loss_tower.append(rec_loss_task)
          rec_images_tower.append(rec_images_task)
          preds_tower.append(preds_task)

    # Calculate the mean of each gradient.
    grads_tower = self._average_gradients(grads_tower)

    # Calculate means of metrics
    loss_tower, acc_tower, preds_tower, clf_loss_tower, rec_loss_tower, \
        rec_images_tower = self._average_metrics_tower(
            loss_tower, acc_tower, preds_tower, clf_loss_tower,
            rec_loss_tower, rec_images_tower)

    return grads_tower, loss_tower, acc_tower, clf_loss_tower, \
        rec_loss_tower, rec_images_tower, preds_tower

  def build_graph(self, image_size=(None, None, None),
                  num_class=None, n_train_samples=None):
    """Build the graph of CapsNet.

    Args:
      image_size: size of input images, should be 3 dimensional
      num_class: number of class of label
      n_train_samples: number of train samples

    Returns:
      tuple of (global_step, train_graph, inputs, labels, train_op,
                saver, summary_op, loss, accuracy, classifier_loss,
                reconstruct_loss, reconstructed_images)
    """
    tf.reset_default_graph()
    train_graph = tf.Graph()

    with train_graph.as_default(), tf.device('/cpu:0'):

      # Get inputs tensor
      inputs, labels, is_training = self._get_inputs(image_size, num_class)

      # Global step
      global_step = tf.placeholder(tf.int16, name='global_step')

      # Optimizer
      optimizer = self._optimizer(self.cfg.OPTIMIZER,
                                  n_train_samples=n_train_samples,
                                  global_step=global_step)

      # Split data for each tower
      x_splits_tower = tf.split(
          axis=0, num_or_size_splits=self.cfg.GPU_NUMBER, value=inputs)
      y_splits_tower = tf.split(
          axis=0, num_or_size_splits=self.cfg.GPU_NUMBER, value=labels)

      # Calculate the gradients for each models tower.
      grads_all, loss_all, acc_all, clf_loss_all, \
          rec_loss_all, rec_images_all, preds_all = \
          [], [], [], [], [], [], []
      for i in range(self.cfg.GPU_NUMBER):
        with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
              grads_tower, loss_tower, acc_tower, clf_loss_tower, \
                  rec_loss_tower, rec_images_tower, preds_tower = \
                  self._calc_on_gpu(
                      i, x_splits_tower, y_splits_tower,
                      image_size, is_training, optimizer)

              # Keep track of the gradients across all towers.
              grads_all.append(grads_tower)

              # Collect metrics of each tower
              loss_all.append(loss_tower)
              acc_all.append(acc_tower)
              clf_loss_all.append(clf_loss_tower)
              rec_loss_all.append(rec_loss_tower)
              rec_images_all.append(rec_images_tower)
              preds_all.append(preds_tower)

      # Calculate the mean of each gradient.
      grads = self._average_gradients(grads_all)

      # Calculate means of metrics
      loss, accuracy, preds, classifier_loss, reconstruct_loss, \
          reconstructed_images = self._average_metrics(
              loss_all, acc_all, preds_all, clf_loss_all,
              rec_loss_all, rec_images_all)

      # Apply the gradients to adjust the shared variables.
      apply_gradient_op = optimizer.apply_gradients(grads)

      # Track the moving averages of all trainable variables.
      variable_averages = tf.train.ExponentialMovingAverage(
          self.cfg.MOVING_AVERAGE_DECAY)
      variables_averages_op = variable_averages.apply(
          tf.trainable_variables())

      # Group all updates to into a single train op.
      train_op = tf.group(apply_gradient_op, variables_averages_op)

      # Create a saver.
      saver = tf.train.Saver(tf.global_variables(),
                             max_to_keep=self.cfg.MAX_TO_KEEP_CKP)

      # Build the summary operation from the last tower summaries.
      tf.summary.scalar('accuracy', accuracy)
      tf.summary.scalar('loss', loss)
      if self.cfg.WITH_REC:
        tf.summary.scalar('clf_loss', classifier_loss)
        tf.summary.scalar('rec_loss', reconstruct_loss)
      summary_op = tf.summary.merge_all()

      return global_step, train_graph, inputs, labels, is_training, \
          train_op, saver, summary_op, loss, accuracy, classifier_loss, \
          reconstruct_loss, reconstructed_images
