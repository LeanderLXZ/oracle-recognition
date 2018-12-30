from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.capsNet import CapsNet


class CapsNetDistribute(CapsNet):

  def __init__(self, cfg, model_arch):
    super(CapsNetDistribute, self).__init__(cfg, model_arch)
    self.clf_arch_info = None
    self.rec_arch_info = None
    self.batch_size = cfg.BATCH_SIZE // cfg.GPU_NUMBER

  def _get_loss(self, inputs, labels, image_size, is_training=None):
    """Calculate the loss running the models.

    Args:
      inputs: inputs. 4D tensor
        - shape:  (batch_size, *image_size)
      labels: labels. 1D tensor of shape [batch_size]
      image_size: size of input images, should be 3 dimensional
      is_training: Whether or not the model is in training mode.

    Returns:
      Tuple: (loss, classifier_loss,
              reconstruct_loss, reconstructed_images)
    """
    # Build inference Graph.
    logits, accuracy, preds = self._inference(
        inputs, labels, is_training=is_training)

    # Calculating the loss.
    loss, classifier_loss, reconstruct_loss, reconstructed_images = \
        self._total_loss(
            inputs, logits, labels, image_size, is_training=is_training)

    return loss, accuracy, preds, classifier_loss, \
        reconstruct_loss, reconstructed_images

  @staticmethod
  def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    This function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
                   is over individual gradients. The inner list is over the
                   gradient calculation for each tower.
        - shape: [[(grad0_gpu0, var0_gpu0), ..., (gradM_gpu0, varM_gpu0)],
                   ...,
                  [(grad0_gpuN, var0_gpuN), ..., (gradM_gpuN, varM_gpuN)]]

    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Each grad_and_vars looks like:
      # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for grad, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_grad = tf.expand_dims(grad, 0)
        # Append on a 'tower' dimension which we will average over.
        grads.append(expanded_grad)

      # grads: [[grad0_gpu0], [grad0_gpu1], ..., [grad0_gpuN]]
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # The Variables are redundant because they are shared across towers.
      # So we will just return the first tower's pointer to the Variable.
      v = grad_and_vars[0][1]  # varI_gpu0
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)

    # average_grads: [(grad0, var0), (grad1, var1), ..., (gradM, varM)]
    return average_grads

  def _average_metrics(self, loss_all, acc_all, preds_all,
                       clf_loss_all, rec_loss_all, rec_images_all):
    """Calculate average of metrics.

    Args:
      loss_all: final losses of each tower, list
      acc_all: accuracies of each tower, list
      preds_all: predictions of each tower, list
      clf_loss_all: classifier losses of each tower, list
      rec_loss_all: reconstruction losses of each tower, list
      rec_images_all: reconstructed images of each tower, list of 4D tensor

    Returns:
      tuple of metrics
    """
    n_tower = float(len(loss_all))

    loss = tf.divide(
        tf.add_n(loss_all), n_tower, name='total_loss')
    assert loss.get_shape() == ()

    accuracy = tf.divide(
        tf.add_n(acc_all), n_tower, name='total_acc')
    assert accuracy.get_shape() == ()

    preds = tf.concat(preds_all, axis=0, name='total_preds')
    assert preds.get_shape()[0] == self.cfg.BATCH_SIZE

    if self.cfg.WITH_REC:
      classifier_loss = tf.divide(
          tf.add_n(clf_loss_all), n_tower, name='total_clf_loss')
      assert classifier_loss.get_shape() == ()

      reconstruct_loss = tf.divide(
          tf.add_n(rec_loss_all), n_tower, name='total_rec_loss')
      assert reconstruct_loss.get_shape() == ()

      reconstructed_images = tf.concat(
          rec_images_all, axis=0, name='total_rec_images')
      assert reconstructed_images.get_shape() == (
        self.cfg.BATCH_SIZE, *rec_images_all[0].get_shape().as_list()[1:])
    else:
      classifier_loss, reconstruct_loss, \
          reconstructed_images = None, None, None

    return loss, accuracy, preds, classifier_loss, \
        reconstruct_loss, reconstructed_images

  def _calc_on_gpu(self, gpu_idx, x_tower, y_tower,
                   image_size, is_training, optimizer):

    # Calculate the loss for one tower.
    loss_tower, acc_tower, preds_tower, clf_loss_tower, rec_loss_tower, \
        rec_images_tower = self._get_loss(
            x_tower, y_tower, image_size, is_training=is_training)

    # Calculate the gradients on this tower.
    grads_tower = optimizer.compute_gradients(loss_tower)

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

        # Dequeues one batch for the GPU
        x_tower, y_tower = x_splits_tower[i], y_splits_tower[i]

        with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):

              grads_tower, loss_tower, acc_tower, clf_loss_tower, \
                  rec_loss_tower, rec_images_tower, preds_tower = \
                  self._calc_on_gpu(i, x_tower, y_tower,
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
      if self.cfg.MOVING_AVERAGE_DECAY:
        variable_averages = tf.train.ExponentialMovingAverage(
            self.cfg.MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)
      else:
        train_op = apply_gradient_op

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
