from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ActivationFunc(object):

  @staticmethod
  def squash(x, batch_size, epsilon):
    """Squashing function

    Args:
      x: A tensor with shape: (batch_size, num_caps, vec_dim, 1).
      batch_size: Batch size
      epsilon: Add epsilon(a very small number) to zeros

    Returns:
      A tensor with the same shape as input tensor but squashed in 'vec_dim'
      dimension.
    """
    vec_shape = x.get_shape().as_list()
    num_caps = vec_shape[1]
    vec_dim = vec_shape[2]

    vec_squared_norm = tf.reduce_sum(tf.square(x), -2, keep_dims=True)
    assert vec_squared_norm.get_shape() == (batch_size, num_caps, 1, 1)

    scalar_factor = tf.div(vec_squared_norm, 1 + vec_squared_norm)
    assert scalar_factor.get_shape() == (batch_size, num_caps, 1, 1)

    unit_vec = tf.div(x, tf.sqrt(vec_squared_norm + epsilon))
    assert unit_vec.get_shape() == (batch_size, num_caps, vec_dim, 1)

    squashed_vec = tf.multiply(scalar_factor, unit_vec)
    assert squashed_vec.get_shape() == (batch_size, num_caps, vec_dim, 1)

    return squashed_vec
