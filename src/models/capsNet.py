from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from models import utils


class CapsNet(object):

  def __init__(self, cfg, model_arch):

    self.cfg = cfg
    self.batch_size = cfg.BATCH_SIZE
    self.classifier = model_arch['classifier']
    self.decoder = model_arch['decoder']
    self.clf_arch_info = None
    self.rec_arch_info = None

  def _get_inputs(self, image_size, num_class):
    """Get input tensors.

    Args:
      image_size: the size of input images, should be 3 dimensional
      num_class: number of class of label
    Returns:
      input tensors
    """
    _inputs = tf.placeholder(
        tf.float32, shape=[self.cfg.BATCH_SIZE, *image_size], name='inputs')
    _labels = tf.placeholder(
        tf.float32, shape=[self.cfg.BATCH_SIZE, num_class], name='labels')
    _is_training = tf.placeholder(tf.bool, name='is_training')

    return _inputs, _labels, _is_training

  def _optimizer(self,
                 opt_name='adam',
                 n_train_samples=None,
                 global_step=None):
    """Optimizer."""
    # Learning rate with exponential decay
    if self.cfg.LR_DECAY:
      learning_rate_ = tf.train.exponential_decay(
          learning_rate=self.cfg.LEARNING_RATE,
          global_step=global_step,
          decay_steps=self.cfg.LR_DECAY_STEPS,
          decay_rate=self.cfg.LR_DECAY_RATE)
      learning_rate_ = tf.maximum(learning_rate_, 1e-6)
    else:
      learning_rate_ = self.cfg.LEARNING_RATE

    if opt_name == 'adam':
      return tf.train.AdamOptimizer(learning_rate_)

    elif opt_name == 'momentum':
      n_batches_per_epoch = \
          n_train_samples // self.cfg.GPU_BATCH_SIZE * self.cfg.GPU_NUMBER
      boundaries = [
          n_batches_per_epoch * x
          for x in np.array(self.cfg.LR_BOUNDARIES, dtype=np.int64)]
      staged_lr = [self.cfg.LEARNING_RATE * x
                   for x in self.cfg.LR_STAGE]
      learning_rate = tf.train.piecewise_constant(
          global_step,
          boundaries, staged_lr)
      return tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=self.cfg.MOMENTUM)

    elif opt_name == 'gd':
      return tf.train.GradientDescentOptimizer(learning_rate_)

    else:
      raise ValueError('Wrong optimizer name!')

  def _margin_loss(self,
                   logits,
                   labels,
                   m_plus=0.9,
                   m_minus=0.1,
                   lambda_=0.5):
    """Calculate margin loss according to Hinton's paper.

    L = T_c * max(0, m_plus-||v_c||)^2 +
        lambda_ * (1-T_c) * max(0, ||v_c||-m_minus)^2

    Args:
      logits: output tensor of capsule layers.
        - shape: (batch_size, num_caps, vec_dim)
      labels: labels
        - shape: (batch_size, num_caps)
      m_plus: truncation of positive item
      m_minus: truncation of negative item
      lambda_: lambda

    Returns:
      margin loss
    """
    logits_shape = logits.get_shape()
    num_caps = logits_shape[1]

    max_square_plus = tf.square(tf.maximum(
        0., m_plus - utils.get_vec_length(
            logits, self.batch_size, self.cfg.EPSILON)))
    max_square_minus = tf.square(tf.maximum(
        0., utils.get_vec_length(
            logits, self.batch_size, self.cfg.EPSILON) - m_minus))
    # max_square_plus & max_plus shape: (batch_size, num_caps)
    assert max_square_plus.get_shape() == (self.batch_size, num_caps)

    loss_c = tf.multiply(labels, max_square_plus) + \
        lambda_ * tf.multiply((1 - labels), max_square_minus)

    # Total margin loss
    margin_loss = tf.reduce_mean(tf.reduce_sum(loss_c, axis=1))

    return margin_loss

  def _margin_loss_h(self, logits, labels, margin=0.4, down_weight=0.5):
    """Penalizes deviations from margin for each logit.

    Each wrong logit costs its distance to margin. For negative logits margin is
    0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.

    Args:
      labels: tensor, one hot encoding of ground truth.
      logits: tensor, model predictions vectors.
      margin: scalar, the margin after subtracting 0.5 from raw_logits.
      down_weight: scalar, the factor for negative cost.

    Returns:
      A scalar with cost for all data point.
    """
    logits = utils.get_vec_length(
        logits, self.batch_size, self.cfg.EPSILON) - 0.5
    positive_cost = labels * tf.cast(tf.less(logits, margin),
                                     tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    loss_c = 0.5 * positive_cost + down_weight * 0.5 * negative_cost
    margin_loss = tf.reduce_mean(tf.reduce_sum(loss_c, axis=1))
    return margin_loss

  def _reconstruct_layers(self, inputs, labels, is_training=None):
    """Reconstruction layer

    Args:
      inputs: input tensor
        - shape: (batch_size, n_class, vec_dim_j)
      labels: labels
        - shape: (batch_size, n_class)
      is_training: Whether or not the model is in training mode.

    Returns:
      output tensor of reconstruction layer
    """
    with tf.variable_scope('masking'):
      # _masked shape: (batch_size, vec_dim_j)
      _masked = tf.reduce_sum(
          tf.multiply(inputs, tf.expand_dims(labels, axis=-1)), axis=1)

    with tf.variable_scope('decoder'):
      # _reconstructed shape: (batch_size, image_size*image_size)
      _reconstructed, self.rec_arch_info = self.decoder(
          _masked, self.cfg, batch_size=self.batch_size,
          is_training=is_training)

    return _reconstructed

  def _loss_without_rec(self, logits, labels):
    """Calculate loss without reconstruction.

    Args:
      logits: output tensor of models
        - shape (batch_size, num_caps, vec_dim)
      labels: labels

    Return:
      total loss
    """
    if self.cfg.CLF_LOSS == 'margin':
      loss = self._margin_loss(
          logits, labels, **self.cfg.MARGIN_LOSS_PARAMS)
    elif self.cfg.CLF_LOSS == 'margin_h':
      loss = self._margin_loss_h(
          logits, labels, **self.cfg.MARGIN_LOSS_H_PARAMS)
    else:
      raise ValueError('Wrong CLF_LOSS Name!')

    return loss

  def _loss_with_rec(self, inputs, logits,
                     labels, image_size, is_training=None):
    """Calculate loss with reconstruction.

    Args:
      inputs: input tensor
        - shape (batch_size, *image_size)
      logits: output tensor of models
        - shape (batch_size, num_caps, vec_dim)
      labels: labels
      image_size: size of image, 3D
      is_training: Whether or not the model is in training mode.

    Return:
      Total loss
    """
    # Reconstruction layers
    # reconstructed shape: (batch_size, image_size*image_size)
    reconstructed = self._reconstruct_layers(
        logits, labels, is_training=is_training)
    if self.cfg.SHOW_TRAINING_DETAILS:
      reconstructed = tf.Print(
          reconstructed, [tf.constant(4)],
          message="\nRECONSTRUCTION layers passed...")

    # Reconstruction loss
    if self.cfg.REC_LOSS == 'mse':
      inputs_flatten = tf.contrib.layers.flatten(inputs)
      if self.cfg.DECODER_TYPE != 'fc':
        reconstructed_ = tf.contrib.layers.flatten(reconstructed)
      else:
        reconstructed_ = reconstructed
      reconstruct_loss = tf.reduce_mean(
          tf.square(reconstructed_ - inputs_flatten))
      reconstructed_images_ = reconstructed
    elif self.cfg.REC_LOSS == 'ce':
      if self.cfg.DECODER_TYPE == 'fc':
        inputs_ = tf.contrib.layers.flatten(inputs)
      else:
        inputs_ = inputs
      reconstruct_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=inputs_, logits=reconstructed))
      reconstructed_images_ = tf.nn.sigmoid(reconstructed)
    else:
      raise ValueError('Wrong reconstruction loss type!')
    reconstruct_loss = tf.identity(reconstruct_loss, name='rec_loss')
    reconstructed_images = tf.reshape(
        reconstructed_images_, shape=[-1, *image_size], name='rec_images')

    # Classifier loss
    if self.cfg.CLF_LOSS == 'margin':
      classifier_loss = self._margin_loss(
          logits, labels, **self.cfg.MARGIN_LOSS_PARAMS)
    elif self.cfg.CLF_LOSS == 'margin_h':
      classifier_loss = self._margin_loss_h(
          logits, labels, **self.cfg.MARGIN_LOSS_H_PARAMS)
    else:
      raise ValueError('Wrong CLF_LOSS Name!')
    classifier_loss = tf.identity(classifier_loss, name='clf_loss')

    loss = classifier_loss + \
        self.cfg.REC_LOSS_SCALE * reconstruct_loss

    if self.cfg.SHOW_TRAINING_DETAILS:
      loss = tf.Print(loss, [tf.constant(5)], message="\nloss calculated...")

    return loss, classifier_loss, reconstruct_loss, reconstructed_images

  def _total_loss(self, inputs, logits, labels, image_size, is_training=None):
    """Get Losses and reconstructed images tensor."""
    if self.cfg.WITH_REC:
      loss, classifier_loss, reconstruct_loss, reconstructed_images = \
          self._loss_with_rec(
              inputs, logits, labels, image_size, is_training=is_training)
    else:
      loss = self._loss_without_rec(logits, labels)
      classifier_loss, reconstruct_loss, reconstructed_images = \
          None, None, None

    loss = tf.identity(loss, name='loss')

    return loss, classifier_loss, reconstruct_loss, reconstructed_images

  def _inference(self, inputs, labels, is_training=None):
    """Build inference graph.

    Args:
      inputs: input tensor
        - shape (batch_size, *image_size)
      labels: labels tensor
      is_training: Whether or not the model is in training mode.

    Return:
      logits: output tensor of models
        - shape: (batch_size, num_caps, vec_dim)
    """
    logits, self.clf_arch_info = self.classifier(
        inputs, self.cfg, self.batch_size, is_training=is_training)

    # Logits shape: (batch_size, num_caps, vec_dim, 1)
    logits = tf.squeeze(logits, name='logits')
    if self.cfg.SHOW_TRAINING_DETAILS:
      logits = tf.Print(logits, [tf.constant(3)],
                        message="\nCAPSULE layers passed...")

    # Predictions
    preds = utils.get_vec_length(logits, self.batch_size, self.cfg.EPSILON)
    preds = tf.identity(preds, name='preds')

    # Accuracy
    correct_pred = tf.equal(tf.argmax(preds, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(
        correct_pred, tf.float32), name='accuracy')

    return logits, accuracy, preds

  def build_graph(self,
                  image_size=(None, None, None),
                  num_class=None,
                  n_train_samples=None):
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

    with train_graph.as_default():

      # Get input placeholders
      inputs, labels, is_training = self._get_inputs(image_size, num_class)

      # Global step
      global_step = tf.placeholder(tf.int16, name='global_step')

      # Optimizer
      optimizer = self._optimizer(opt_name=self.cfg.OPTIMIZER,
                                  n_train_samples=n_train_samples,
                                  global_step=global_step)

      # Build inference Graph
      logits, accuracy, preds = self._inference(
          inputs, labels, is_training=is_training)

      # Build reconstruction part
      loss, classifier_loss, reconstruct_loss, reconstructed_images = \
          self._total_loss(
              inputs, logits, labels, image_size, is_training=is_training)

      # Optimizer
      if self.cfg.SHOW_TRAINING_DETAILS:
        loss = tf.Print(loss, [tf.constant(6)],
                        message="\nUpdating gradients...")
      train_op = optimizer.minimize(loss)

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
