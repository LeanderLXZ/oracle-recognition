from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
slim = tf.contrib.slim

epsilon = 1e-9

from models.caps_activate_fn import ActivationFunc
from models.layers import variable_on_cpu
from models.layers import get_act_fn


class CapsLayer(object):

  def __init__(self,
               cfg,
               num_caps=None,
               vec_dim=None,
               route_epoch=None,
               batch_size=None,
               share_weights=False,
               idx=0):
    """Initialize capsule layer.

    Args:
      cfg: configuration
      num_caps: number of capsules of this layer
      vec_dim: dimensions of vectors of capsules
      route_epoch: number of dynamic routing iteration
      batch_size: number of samples per batch
      idx: index of layer
    """
    self.cfg = cfg
    self.num_caps = num_caps
    self.vec_dim = vec_dim
    self.route_epoch = route_epoch
    self.batch_size = batch_size
    self.idx = idx
    self.tensor_shape = None
    self.share_weights = share_weights

  @property
  def params(self):
    """Parameters of this layer."""
    return {
      'num_caps': self.num_caps,
      'vec_dim': self.vec_dim,
      'route_epoch': self.route_epoch,
      'batch_size': self.batch_size,
      'share_weights': self.share_weights,
      'idx': self.idx
    }

  def __call__(self, inputs):
    """Apply dynamic routing.

    Args:
      inputs: input tensor
        - shape: (batch_size, num_caps_i, vec_dim_i, 1)

    Returns:
      output tensor
        - shape (batch_size, num_caps_j, vec_dim_j, 1)
    """
    with tf.variable_scope('caps_{}'.format(self.idx)):
      self.v_j = self.dynamic_routing(
          inputs, self.num_caps, self.vec_dim,
          self.route_epoch, self.batch_size)

    self.tensor_shape = self.v_j.get_shape().as_list()
    return self.v_j

  def dynamic_routing(self, inputs, num_caps_j,
                      vec_dim_j, route_epoch, batch_size):
    """Dynamic routing according to Hinton's paper.

    Args:
      inputs: input tensor
        - shape: (batch_size, num_caps_i, vec_dim_i, 1)
      num_caps_j: number of capsules of upper layer
      vec_dim_j: dimensions of vectors of upper layer
      route_epoch: number of dynamic routing iteration
      batch_size: number of samples per batch

    Returns:
      output tensor
        - shape (batch_size, num_caps_j, vec_dim_j, 1)
    """
    inputs_shape = inputs.get_shape().as_list()
    num_caps_i = inputs_shape[1]
    vec_dim_i = inputs_shape[2]
    v_j = None

    # Reshape input tensor
    inputs_shape_new = [batch_size, num_caps_i, 1, vec_dim_i, 1]
    inputs = tf.reshape(inputs, shape=inputs_shape_new)
    inputs = tf.tile(inputs, [1, 1, num_caps_j, 1, 1], name='input_tensor')
    # inputs shape: (batch_size, num_caps_i, num_caps_j, vec_dim_i, 1)
    assert inputs.get_shape() == (
        batch_size, num_caps_i, num_caps_j, vec_dim_i, 1)

    # Initializing weights
    if self.share_weights:
      weights_shape = [1, 1, num_caps_j, vec_dim_j, vec_dim_i]
    else:
      weights_shape = [1, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i]
    # Reuse weights
    if self.cfg.VAR_ON_CPU:
      weights = variable_on_cpu(
          name='weights',
          shape=weights_shape,
          initializer=tf.truncated_normal_initializer(
              stddev=self.cfg.WEIGHTS_STDDEV, dtype=tf.float32),
          dtype=tf.float32)
    else:
      weights = tf.get_variable(
          name='weights',
          shape=weights_shape,
          initializer=tf.truncated_normal_initializer(
              stddev=self.cfg.WEIGHTS_STDDEV, dtype=tf.float32),
          dtype=tf.float32)
    weights = tf.tile(weights, [batch_size, 1, 1, 1, 1])
    if self.share_weights:
      weights = tf.tile(weights, [1, num_caps_i, 1, 1, 1])
    # weights shape: (batch_size, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i)
    assert weights.get_shape() == (
        batch_size, num_caps_i, num_caps_j, vec_dim_j, vec_dim_i)

    # Calculating u_hat
    # ( , , , vec_dim_j, vec_dim_i) x ( , , , vec_dim_i, 1)
    # -> ( , , , vec_dim_j, 1) -> squeeze -> ( , , , vec_dim_j)
    u_hat = tf.matmul(weights, inputs, name='u_hat')
    # u_hat shape: (batch_size, num_caps_i, num_caps_j, vec_dim_j, 1)
    assert u_hat.get_shape() == (
        batch_size, num_caps_i, num_caps_j, vec_dim_j, 1)

    # u_hat_stop
    # Do not transfer the gradient of u_hat_stop during back-propagation
    u_hat_stop = tf.stop_gradient(u_hat, name='u_hat_stop')

    # Initializing b_ij
    if self.cfg.VAR_ON_CPU:
      b_ij = variable_on_cpu(
          name='b_ij',
          shape=[batch_size, num_caps_i, num_caps_j, 1, 1],
          initializer=tf.zeros_initializer(),
          dtype=tf.float32,
          trainable=False)
    else:
      b_ij = tf.get_variable(
          name='b_ij',
          shape=[batch_size, num_caps_i, num_caps_j, 1, 1],
          initializer=tf.zeros_initializer(),
          dtype=tf.float32,
          trainable=False)
    # b_ij shape: (batch_size, num_caps_i, num_caps_j, 1, 1)
    assert b_ij.get_shape() == (
        batch_size, num_caps_i, num_caps_j, 1, 1)

    def _sum_and_activate(_u_hat, _c_ij, cfg_, name=None):
      """Get sum of vectors and apply activation function."""
      # Calculating s_j(using u_hat)
      # Using u_hat but not u_hat_stop in order to transfer gradients.
      _s_j = tf.reduce_sum(tf.multiply(_u_hat, _c_ij), axis=1)
      # _s_j shape: (batch_size, num_caps_j, vec_dim_j, 1)
      assert _s_j.get_shape() == (
          batch_size, num_caps_j, vec_dim_j, 1)

      # Applying Squashing
      _v_j = ActivationFunc.squash(_s_j, batch_size, cfg_.EPSILON)
      # _v_j shape: (batch_size, num_caps_j, vec_dim_j, 1)
      assert _v_j.get_shape() == (
          batch_size, num_caps_j, vec_dim_j, 1)

      _v_j = tf.identity(_v_j, name=name)

      return _v_j

    for iter_route in range(route_epoch):

      with tf.variable_scope('iter_route_{}'.format(iter_route)):

        # Calculate c_ij for every epoch
        c_ij = tf.nn.softmax(b_ij, dim=2)

        # c_ij shape: (batch_size, num_caps_i, num_caps_j, 1, 1)
        assert c_ij.get_shape() == (
            batch_size, num_caps_i, num_caps_j, 1, 1)

        # Applying back-propagation at last epoch.
        if iter_route == route_epoch - 1:
          # c_ij_stop
          # Do not transfer the gradient of c_ij_stop during back-propagation.
          c_ij_stop = tf.stop_gradient(c_ij, name='c_ij_stop')

          # Calculating s_j(using u_hat) and Applying activation function.
          # Using u_hat but not u_hat_stop in order to transfer gradients.
          v_j = _sum_and_activate(
              u_hat, c_ij_stop, self.cfg, name='v_j')

        # Do not apply back-propagation if it is not last epoch.
        else:
          # Calculating s_j(using u_hat_stop) and Applying activation function.
          # Using u_hat_stop so that the gradient will not be transferred to
          # routing processes.
          v_j = _sum_and_activate(
              u_hat_stop, c_ij, self.cfg, name='v_j')

          # Updating: b_ij <- b_ij + vj x u_ij
          v_j_reshaped = tf.reshape(
              v_j, shape=[-1, 1, num_caps_j, 1, vec_dim_j])
          v_j_reshaped = tf.tile(
              v_j_reshaped,
              [1, num_caps_i, 1, 1, 1],
              name='v_j_reshaped')
          # v_j_reshaped shape:
          # (batch_size, num_caps_i, num_caps_j, 1, vec_dim_j)
          assert v_j_reshaped.get_shape() == (
              batch_size, num_caps_i, num_caps_j, 1, vec_dim_j)

          # ( , , , 1, vec_dim_j) x ( , , , vec_dim_j, 1)
          # -> squeeze -> (batch_size, num_caps_i, num_caps_j, 1, 1)
          delta_b_ij = tf.matmul(
              v_j_reshaped, u_hat_stop, name='delta_b_ij')
          # delta_b_ij shape: (batch_size, num_caps_i, num_caps_j, 1)
          assert delta_b_ij.get_shape() == (
              batch_size, num_caps_i, num_caps_j, 1, 1)

          b_ij = tf.add(b_ij, delta_b_ij, name='b_ij')
          # b_ij shape: (batch_size, num_caps_i, num_caps_j, 1, 1)
          assert b_ij.get_shape() == (
              batch_size, num_caps_i, num_caps_j, 1, 1)

    # v_j_out shape: (batch_size, num_caps_j, vec_dim_j, 1)
    assert v_j.get_shape() == (
        batch_size, num_caps_j, vec_dim_j, 1)

    return v_j


class Conv2CapsLayer(object):

  def __init__(self,
               cfg,
               kernel_size=None,
               stride=None,
               n_kernel=None,
               vec_dim=None,
               padding='SAME',
               act_fn='relu',
               w_init_fn=tf.contrib.layers.xavier_initializer(),
               use_bias=True,
               batch_size=None):
    """Generate a Capsule layer using convolution kernel.

    Args:
      cfg: configuration
      kernel_size: size of convolution kernel
      stride: stride of convolution kernel
      n_kernel: depth of convolution kernel
      padding: padding type of convolution kernel
      act_fn: activation function of convolution layer
      vec_dim: dimensions of vectors of capsule
      w_init_fn: weights initializer of convolution layer
      use_bias: add biases
      batch_size: number of samples per batch
    """
    self.cfg = cfg
    self.kernel_size = kernel_size
    self.stride = stride
    self.n_kernel = n_kernel
    self.padding = padding
    self.act_fn = act_fn
    self.vec_dim = vec_dim
    self.w_init_fn = w_init_fn
    self.use_bias = use_bias
    self.batch_size = batch_size
    self.tensor_shape = None

  @property
  def params(self):
    """Parameters of this layer."""
    return {
      'kernel_size': self.kernel_size,
      'stride': self.stride,
      'n_kernel': self.n_kernel,
      'padding': self.padding,
      'act_fn': self.act_fn,
      'vec_dim': self.vec_dim,
      'w_init_fn': self.w_init_fn,
      'use_bias': self.use_bias,
      'batch_size': self.batch_size
    }

  def __call__(self, inputs):
    """Convert a convolution layer to capsule layer.

    Args:
      inputs: input tensor
        - shape: (batch_size, height, width, depth)

    Returns:
      tensor of capsules
        - shape: (batch_size, num_caps_j, vec_dim_j, 1)
    """
    with tf.variable_scope('conv2caps'):
      # Convolution layer
      activation_fn = get_act_fn(self.act_fn)

      if self.cfg.VAR_ON_CPU:
        kernels = variable_on_cpu(
            name='kernels',
            shape=[self.kernel_size, self.kernel_size,
                   inputs.get_shape().as_list()[3],
                   self.n_kernel * self.vec_dim],
            initializer=self.w_init_fn,
            dtype=tf.float32)
        caps = tf.nn.conv2d(
            input=inputs,
            filter=kernels,
            strides=[1, self.stride, self.stride, 1],
            padding=self.padding)

        if self.use_bias:
          biases = variable_on_cpu(
              name='biases',
              shape=[self.n_kernel * self.vec_dim],
              initializer=tf.zeros_initializer(),
              dtype=tf.float32)
          caps = tf.add(caps, biases)

        if activation_fn is not None:
          caps = activation_fn(caps)

      else:
        biases_initializer = tf.zeros_initializer() if self.use_bias else None
        caps = tf.contrib.layers.conv2d(
            inputs=inputs,
            num_outputs=self.n_kernel * self.vec_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            activation_fn=activation_fn,
            weights_initializer=self.w_init_fn,
            biases_initializer=biases_initializer)

      # Reshape and generating a capsule layer
      # caps shape:
      # (batch_size, img_height, img_width, self.n_kernel * self.vec_dim)
      caps_shape = caps.get_shape().as_list()
      num_capsule = caps_shape[1] * caps_shape[2] * self.n_kernel
      caps = tf.reshape(caps, [self.batch_size, -1, self.vec_dim, 1])
      # caps shape: (batch_size, num_caps_j, vec_dim_j, 1)
      assert caps.get_shape() == (
        self.batch_size, num_capsule, self.vec_dim, 1)

      # Applying activation function
      caps_activated = ActivationFunc.squash(
          caps, self.batch_size, self.cfg.EPSILON)
      # caps_activated shape: (batch_size, num_caps_j, vec_dim_j, 1)
      assert caps_activated.get_shape() == (
        self.batch_size, num_capsule, self.vec_dim, 1)

      self.tensor_shape = caps_activated.get_shape().as_list()
      return caps_activated


class MatrixCapsLayer(object):

  def __init__(self,
               cfg,
               shape=(3, 3, 32, 32),
               strides=(1, 2, 2, 1),
               route_epoch=None,
               idx=0):
    """Initialize Matrix capsule layer.

    Args:
      cfg: configuration
      shape: shape of output
      strides: strides of convolution
      route_epoch: number of dynamic routing iteration
      idx: index of layer
    """
    self.cfg = cfg
    self.shape = shape
    self.strides = strides
    self.route_epoch = route_epoch
    self.idx = idx

  @property
  def params(self):
    """Parameters of this layer."""
    return {
      'shape': self.shape,
      'strides': self.strides,
      'route_epoch': self.route_epoch,
      'idx': self.idx
    }

  def __call__(self, inputs):
    """Apply EM routing.
    """
    with tf.variable_scope('caps_{}'.format(self.idx)):
      self.v_j = self.capsules_conv(
        inputs,
        shape=self.shape,
        strides=self.strides,
        iterations=self.route_epoch,
        name='mcaps_conv_' + str(self.idx)
      )

    self.tensor_shape = self.v_j.get_shape().as_list()
    return self.v_j

  @staticmethod
  def _matmul_broadcast(x, y, name):
    """Compute x @ y, broadcasting over the first `N - 2` ranks.
    """
    with tf.variable_scope(name) as scope:
      return tf.reduce_sum(
        x[..., tf.newaxis] * y[..., tf.newaxis, :, :], axis=-2)

  @staticmethod
  def _get_variable_wrapper(name, shape=None, dtype=None, initializer=None,
                            regularizer=None, trainable=True, collections=None,
                            caching_device=None, partitioner=None,
                            validate_shape=True, custom_getter=None):
    """Wrapper over tf.get_variable().
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(
        name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, trainable=trainable,
        collections=collections, caching_device=caching_device,
        partitioner=partitioner, validate_shape=validate_shape,
        custom_getter=custom_getter
      )
    return var

  def _get_weights_wrapper(self, name, shape, dtype=tf.float32,
                           initializer=initializers.xavier_initializer(),
                           weights_decay_factor=None):
    """Wrapper over _get_variable_wrapper() to get weights,
    with weights decay factor in loss.
    """
    weights = self._get_variable_wrapper(
      name=name, shape=shape, dtype=dtype, initializer=initializer)
    if weights_decay_factor is not None and weights_decay_factor > 0.0:
      weights_wd = tf.multiply(
        tf.nn.l2_loss(weights), weights_decay_factor, name=name + '/l2loss')
      tf.add_to_collection('losses', weights_wd)
    return weights

  def _get_biases_wrapper(self, name, shape, dtype=tf.float32,
                          initializer=tf.constant_initializer(0.0)):
    """Wrapper over _get_variable_wrapper() to get bias.
    """
    return self._get_variable_wrapper(
      name=name, shape=shape, dtype=dtype, initializer=initializer)

  def _conv2d_wrapper(self, inputs, shape, strides,
                      padding, add_bias, activation_fn, name):
    """Wrapper over tf.nn.conv2d().
    """

    with tf.variable_scope(name):
      kernel = self._get_weights_wrapper(
        name='weights', shape=shape, weights_decay_factor=0.0)
      output = tf.nn.conv2d(
        inputs, filter=kernel, strides=strides, padding=padding, name='conv')
      if add_bias:
        biases = self._get_biases_wrapper(name='biases', shape=[shape[-1]])
        output = tf.add(output, biases, name='biasAdd')
      if activation_fn is not None:
        output = activation_fn(output, name='activation')
    return output

  def _separable_conv2d_wrapper(self, inputs, depthwise_shape, pointwise_shape,
                                strides, padding, add_bias, activation_fn,
                                name):
    """Wrapper over tf.nn.separable_conv2d().
    """

    with tf.variable_scope(name):
      dkernel = self._get_weights_wrapper(
        name='depthwise_weights', shape=depthwise_shape,
        weights_decay_factor=0.0
      )
      pkernel = self._get_weights_wrapper(
        name='pointwise_weights', shape=pointwise_shape,
        weights_decay_factor=0.0
      )
      output = tf.nn.separable_conv2d(
        input=inputs, depthwise_filter=dkernel, pointwise_filter=pkernel,
        strides=strides, padding=padding, name='conv'
      )
      if add_bias:
        biases = self._get_biases_wrapper(
          name='biases', shape=[pointwise_shape[-1]]
        )
        output = tf.add(
          output, biases, name='biasAdd'
        )
      if activation_fn is not None:
        output = activation_fn(
          output, name='activation'
        )

    return output

  def _depthwise_conv2d_wrapper(self, inputs, shape, strides, padding, add_bias,
                                activation_fn, name):
    """Wrapper over tf.nn.depthwise_conv2d().
    """

    with tf.variable_scope(name):
      dkernel = self._get_weights_wrapper(
        name='depthwise_weights', shape=shape, weights_decay_factor=0.0
      )
      output = tf.nn.depthwise_conv2d(
        inputs, filter=dkernel, strides=strides, padding=padding, name='conv'
      )
      if add_bias:
        d_ = output.get_shape()[-1].value
        biases = self._get_biases_wrapper(
          name='biases', shape=[d_]
        )
        output = tf.add(
          output, biases, name='biasAdd'
        )
      if activation_fn is not None:
        output = activation_fn(
          output, name='activation'
        )

    return output

  def capsules_init(self, inputs, shape, strides, padding, pose_shape, name):
    """This constructs a primary capsule layer from a regular convolution layer.

    :param inputs: a regular convolution layer with shape [N, H, W, C],
      where often N is batch_size, H is height, W is width, and C is channel.
    :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
      where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
    :param strides: strides [1, SH, SW, 1] w.r.t [N, H, W, C], often [1, 1, 1, 1], or [1, 2, 2, 1].
    :param padding: padding, often SAME or VALID.
    :param pose_shape: the shape of each pose matrix, [PH, PW],
      where PH is pose height, and PW is pose width.
    :param name: name.

    :return: (poses, activations),
      poses: [N, H, W, C, PH, PW], activations: [N, H, W, C],
      where often N is batch_size, H is output height, W is output width, C is output channels,
      and PH is pose height, and PW is pose width.

    note: with respect to the paper, matrix capsules with EM routing, figure 1,
      this function provides the operation to build from
      ReLU Conv1 [batch_size, 14, 14, A] to
      PrimaryCapsule poses [batch_size, 14, 14, B, 4, 4], activations [batch_size, 14, 14, B] with
      Kernel [A, B, 4 x 4 + 1], specifically,
      weight kernel shape [1, 1, A, B], strides [1, 1, 1, 1], pose_shape [4, 4]
    """

    # assert len(pose_shape) == 2

    with tf.variable_scope(name):
      # poses: build one by one
      # poses = []
      # for ph in xrange(pose_shape[0]):
      #   poses_wire = []
      #   for pw in xrange(pose_shape[1]):
      #     poses_unit = _conv2d_wrapper(
      #       inputs, shape=shape, strides=strides, padding=padding, add_bias=False, activation_fn=None, name=name+'_pose_'+str(ph)+'_'+str(pw)
      #     )
      #     poses_wire.append(poses_unit)
      #   poses.append(tf.stack(poses_wire, axis=-1, name=name+'_poses_'+str(ph)))
      # poses = tf.stack(poses, axis=-1, name=name+'_poses')

      # poses: simplified build all at once
      poses = self._conv2d_wrapper(
        inputs,
        shape=shape[0:-1] + [shape[-1] * pose_shape[0] * pose_shape[1]],
        strides=strides,
        padding=padding,
        add_bias=False,
        activation_fn=None,
        name='pose_stacked'
      )
      # poses = slim.conv2d(
      #   inputs,
      #   num_outputs=shape[-1] * pose_shape[0] * pose_shape[1],
      #   kernel_size=shape[0:2],
      #   stride=strides[1],
      #   padding=padding,
      #   activation_fn=None,
      #   weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04),
      #   scope='poses_stacked'
      # )
      # shape: poses_shape[0:-1] + [shape[-1], pose_shape[0], pose_shape[1]]
      # modified to [-1] + poses_shape[1:-1] + [shape[-1], pose_shape[0], pose_shape[1]]
      # to allow_smaller_final_batch dynamic batch size
      poses_shape = poses.get_shape().as_list()
      poses = tf.reshape(
        poses, shape=[-1] + poses_shape[1:-1] + [shape[-1], pose_shape[0],
                                                 pose_shape[1]], name='poses'
      )

      activations = self._conv2d_wrapper(
        inputs,
        shape=shape,
        strides=strides,
        padding=padding,
        add_bias=False,
        activation_fn=tf.sigmoid,
        name='activation'
      )
      # activations = slim.conv2d(
      #   inputs,
      #   num_outputs=shape[-1],
      #   kernel_size=shape[0:2],
      #   stride=strides[1],
      #   padding=padding,
      #   activation_fn=tf.sigmoid,
      #   weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04),
      #   scope='activations'
      # )
      # activations = tf.Print(
      #   activations, [activations.shape, activations[0, 4:7, 5:8, :]], str(activations.name) + ':', summarize=20
      # )

      # add into GraphKeys.SUMMARIES
      tf.summary.histogram(
        'activations', activations
      )

    return poses, activations

  def capsules_conv(self, inputs, shape, strides, iterations, name):
    """This constructs a convolution capsule layer from a primary or convolution capsule layer.

    :param inputs: a primary or convolution capsule layer with poses and activations,
      poses shape [N, H, W, C, PH, PW], activations shape [N, H, W, C]
    :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
      where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
    :param strides: strides [1, SH, SW, 1] w.r.t [N, H, W, C], often [1, 1, 1, 1], or [1, 2, 2, 1].
    :param iterations: number of iterations in EM routing, often 3.
    :param name: name.

    :return: (poses, activations) same as capsule_init().

    note: with respect to the paper, matrix capsules with EM routing, figure 1,
      this function provides the operation to build from
      PrimaryCapsule poses [batch_size, 14, 14, B, 4, 4], activations [batch_size, 14, 14, B] to
      ConvCapsule1 poses [batch_size, 6, 6, C, 4, 4], activations [batch_size, 6, 6, C] with
      Kernel [KH=3, KW=3, B, C, 4, 4], specifically,
      weight kernel shape [3, 3, B, C], strides [1, 2, 2, 1], pose_shape [4, 4]

      also, this function provides the operation to build from
      ConvCapsule1 poses [batch_size, 6, 6, C, 4, 4], activations [batch_size, 6, 6, C] to
      ConvCapsule2 poses [batch_size, 4, 4, D, 4, 4], activations [batch_size, 4, 4, D] with
      Kernel [KH=3, KW=3, C, D, 4, 4], specifically,
      weight kernel shape [3, 3, C, D], strides [1, 1, 1, 1], pose_shape [4, 4]
    """

    inputs_poses, inputs_activations = inputs
    inputs_poses_shape = inputs_poses.get_shape().as_list()

    assert shape[2] == inputs_poses_shape[3]
    assert strides[0] == strides[-1] == 1

    # note: with respect to the paper, matrix capsules with EM routing, 1.1 previous work on capsules:
    # 3. it uses a vector of length n rather than a matrix with n elements to represent a pose, so its transformation matrices have n^2 parameters rather than just n.

    # this explicit express a matrix PH x PW should be use as a viewpoint transformation matrix to adjust pose.

    with tf.variable_scope(name):
      # kernel: [KH, KW, I, O, PW, PW]
      # yg note: if pose is irregular such as 5x3, then kernel for pose view transformation should be 3x3.
      kernel = self._get_weights_wrapper(
        name='pose_view_transform_weights',
        shape=shape + [inputs_poses_shape[-1], inputs_poses_shape[-1]]
      )

      # note: https://github.com/tensorflow/tensorflow/issues/216
      # tf.matmul doesn't support for broadcasting at this moment, work around with _matmul_broadcast().
      # construct conv patches (this should be a c++ dedicated function support for capsule convolution)
      hk_offsets = [
        [(h_offset + k_offset) for k_offset in range(0, shape[0])] for h_offset
        in
        range(0, inputs_poses_shape[1] + 1 - shape[0], strides[1])
      ]
      wk_offsets = [
        [(w_offset + k_offset) for k_offset in range(0, shape[1])] for w_offset
        in
        range(0, inputs_poses_shape[2] + 1 - shape[1], strides[2])
      ]

      # inputs_poses [N, H, W, I, PH, PW] patches into [N, OH, OW, KH, KW, I, 1, PH, PW]
      # where OH, OW are output height and width determined by H, W, shape and strides,
      # and KH and KW are kernel height and width determined by shape
      inputs_poses_patches = tf.transpose(
        tf.gather(
          tf.gather(
            inputs_poses, hk_offsets, axis=1, name='gather_poses_height_kernel'
          ), wk_offsets, axis=3, name='gather_poses_width_kernel'
        ), perm=[0, 1, 3, 2, 4, 5, 6, 7], name='inputs_poses_patches'
      )
      # inputs_poses_patches expand dimensions from [N, OH, OW, KH, KW, I, PH, PW] to [N, OH, OW, KH, KW, I, 1, PH, PW]
      inputs_poses_patches = inputs_poses_patches[..., tf.newaxis, :, :]
      # inputs_votes: [N, OH, OW, KH, KW, I, O, PH, PW]
      # inputs_votes should be the inputs_poses_patches multiply with the kernel view transformation matrix
      # temporary workaround with tf.tile.
      inputs_poses_patches = tf.tile(
        inputs_poses_patches, [1, 1, 1, 1, 1, 1, shape[-1], 1, 1],
        name='workaround_broadcasting_issue'
      )
      votes = self._matmul_broadcast(
        inputs_poses_patches, kernel,
        name='inputs_poses_patches_view_transformation'
      )
      votes_shape = votes.get_shape().as_list()
      # inputs_votes: reshape into [N, OH, OW, KH x KW x I, O, PH x PW]
      # votes = tf.reshape(
      #   votes, [
      #     votes_shape[0],  votes_shape[1],  votes_shape[2],
      #     votes_shape[3] * votes_shape[4] * votes_shape[5],
      #     votes_shape[6],  votes_shape[7] * votes_shape[8]
      #   ], name='votes'
      # )
      votes = tf.reshape(
        votes, [
          -1, votes_shape[1], votes_shape[2],
          votes_shape[3] * votes_shape[4] * votes_shape[5],
          votes_shape[6], votes_shape[7] * votes_shape[8]
        ], name='votes'
      )
      # stop gradient on votes
      # votes = tf.stop_gradient(votes, name='votes_stop_gradient')

      # inputs_activations: [N, H, W, I] patches into [N, OH, OW, KH, KW, I]
      inputs_activations_patches = tf.transpose(
        tf.gather(
          tf.gather(
            inputs_activations, hk_offsets, axis=1,
            name='gather_activations_height_kernel'
          ), wk_offsets, axis=3, name='gather_activations_width_kernel'
        ), perm=[0, 1, 3, 2, 4, 5], name='inputs_activations_patches'
      )
      # inputs_activations: [N, OH, OW, KH, KW, I] reshape into [N, OH, OW, KH x KW x I]
      # re-use votes_shape so that make sure the votes and i_activations shape match each other.
      # i_activations = tf.reshape(
      #   inputs_activations_patches, [
      #     votes_shape[0],  votes_shape[1],  votes_shape[2],
      #     votes_shape[3] * votes_shape[4] * votes_shape[5]
      #   ], name='i_activations'
      # )
      i_activations = tf.reshape(
        inputs_activations_patches, [
          -1, votes_shape[1], votes_shape[2],
          votes_shape[3] * votes_shape[4] * votes_shape[5]
        ], name='i_activations'
      )

      # beta_v and beta_a one for each output capsule: [1, 1, 1, O]
      beta_v = self._get_weights_wrapper(
        name='beta_v', shape=[1, 1, 1, votes_shape[6]]
      )
      beta_a = self._get_weights_wrapper(
        name='beta_a', shape=[1, 1, 1, votes_shape[6]]
      )

      # output poses and activations via matrix capsules_em_routing algorithm
      # this operation involves inputs and output capsules across all (hk_offsets, wk_offsets), across all channels
      # poses: [N, OH, OW, O, PH x PW], activations: [N, OH, OW, O]
      poses, activations = self.matrix_capsules_em_routing(
        votes, i_activations, beta_v, beta_a, iterations, name='em_routing'
      )
      # poses: [N, OH, OW, O, PH, PW]
      # poses = tf.reshape(
      #   poses, [
      #     votes_shape[0], votes_shape[1], votes_shape[2], votes_shape[6], votes_shape[7], votes_shape[8]
      #   ]
      # )
      poses = tf.reshape(
        poses, [
          -1, votes_shape[1], votes_shape[2], votes_shape[6], votes_shape[7],
          votes_shape[8]
        ]
      )

      # add into GraphKeys.SUMMARIES
      tf.summary.histogram(
        'activations', activations
      )

    return poses, activations

  def capsules_fc(self, inputs, num_classes, iterations, name):
    """This constructs an output layer from a primary or convolution capsule layer via
      a full-connected operation with one view transformation kernel matrix shared across each channel.

    :param inputs: a primary or convolution capsule layer with poses and activations,
      poses shape [N, H, W, C, PH, PW], activations shape [N, H, W, C]
    :param num_classes: number of classes.
    :param iterations: number of iterations in EM routing, often 3.
    :param name: name.

    :return: (pose, activation) same as capsule_init().

    note: with respect to the paper, matrix capsules with EM routing, figure 1,
      This is the D -> E in figure.
      This step includes two major sub-steps:
        1. Apply one view transform weight matrix PH x PW (4 x 4) to each input channel, this view transform matrix is
          shared across (height, width) locations. This is the reason the kernel labelled in D has 1 x 1, and the reason
          the number of variables of weights is D x E x 4 x 4.
        2. Re-struct the inputs vote from [N, H, W, I, PH, PW] into [N, H x W x I, PH x PW],
          add scaled coordinate on first two elements, EM routing an output [N, O, PH x PW],
          and reshape output [N, O, PH, PW].
      The difference between fully-connected layer and convolution layer, is that:
        1. The corresponding kernel size KH, KW in this fully-connected here is actually the whole H, W, instead of 1, 1.
        2. The view transformation matrix is shared within KH, KW (i.e., H, W) in this fully-connected layer,
          whereas in the convolution capsule layer, the view transformation can be different for each capsule
          in the KH, KW, but shared across different (height, width) locations.
    """

    inputs_poses, inputs_activations = inputs

    inputs_poses_shape = inputs_poses.get_shape().as_list()

    inputs_activations_shape = inputs_activations.get_shape().as_list()

    with tf.variable_scope(name):
      # kernel: [I, O, PW, PW]
      # yg note: if pose is irregular such as 5x3, then kernel for pose view transformation should be 3x3.
      kernel = self._get_weights_wrapper(
        name='pose_view_transform_weights',
        shape=[
          inputs_poses_shape[3], num_classes, inputs_poses_shape[-1],
          inputs_poses_shape[-1]
        ],
      )

      # inputs_pose_expansion: [N, H, W, I, 1, PH, PW]
      # inputs_pose_expansion: expand inputs_pose dimension to match with kernel for broadcasting,
      # share the transformation matrices between different positions of the same capsule type,
      # share the transformation matrices as kernel (1, 1) broadcasting to inputs pose expansion (H, W)
      inputs_poses_expansion = inputs_poses[..., tf.newaxis, :, :]

      # temporary workaround with tf.tile.
      inputs_poses_expansion = tf.tile(
        inputs_poses_expansion, [1, 1, 1, 1, num_classes, 1, 1],
        name='workaround_broadcasting_issue'
      )

      # votes: [N, H, W, I, O, PH, PW]
      votes = self._matmul_broadcast(
        inputs_poses_expansion, kernel, name='votes'
      )
      votes_shape = votes.get_shape().as_list()
      # votes: reshape into [N, H, W, I, O, PH x PW]
      votes = tf.reshape(
        votes, [-1] + votes_shape[1:-2] + [votes_shape[-2] * votes_shape[-1]]
      )
      # stop gradient on votes
      # votes = tf.stop_gradient(votes, name='votes_stop_gradient')

      # add scaled coordinate (row, column) of the center of the receptive field of each capsule
      # to the first two elements of its vote
      height = inputs_poses_shape[1]
      width = inputs_poses_shape[2]

      coordinate_offset_hh = tf.reshape(
        (tf.range(height, dtype=tf.float32) + 0.50) / height, [1, height, 1, 1, 1])
      coordinate_offset_h0 = tf.constant(
        0.0, shape=[1, height, 1, 1, 1], dtype=tf.float32
      )
      coordinate_offset_h = tf.stack(
        [coordinate_offset_hh, coordinate_offset_h0] + [coordinate_offset_h0 for
                                                        _ in range(14)],
        axis=-1
      )

      coordinate_offset_ww = tf.reshape(
        (tf.range(width, dtype=tf.float32) + 0.50) / width, [1, 1, width, 1, 1]
      )
      coordinate_offset_w0 = tf.constant(
        0.0, shape=[1, 1, width, 1, 1], dtype=tf.float32
      )
      coordinate_offset_w = tf.stack(
        [coordinate_offset_w0, coordinate_offset_ww] + [coordinate_offset_w0 for
                                                        _ in range(14)],
        axis=-1
      )

      votes = votes + coordinate_offset_h + coordinate_offset_w

      # votes: reshape into [N, H x W x I, O, PH x PW]
      # votes = tf.reshape(
      #   votes, [
      #     votes_shape[0],
      #     votes_shape[1] * votes_shape[2] * votes_shape[3],
      #     votes_shape[4],  votes_shape[5] * votes_shape[6]
      #   ]
      # )
      votes = tf.reshape(
        votes, [
          -1,
          votes_shape[1] * votes_shape[2] * votes_shape[3],
          votes_shape[4], votes_shape[5] * votes_shape[6]
        ]
      )

      # inputs_activations: [N, H, W, I]
      # inputs_activations: reshape into [N, H x W x I]
      # i_activations = tf.reshape(
      #   inputs_activations, [
      #     inputs_activations_shape[0],
      #     inputs_activations_shape[1] * inputs_activations_shape[2] * inputs_activations_shape[3]
      #   ]
      # )
      i_activations = tf.reshape(
        inputs_activations, [
          -1,
          inputs_activations_shape[1] * inputs_activations_shape[2] *
          inputs_activations_shape[3]
        ]
      )

      # beta_v and beta_a one for each output capsule: [1, O]
      beta_v = self._get_weights_wrapper(
        name='beta_v', shape=[1, num_classes]
      )
      beta_a = self._get_weights_wrapper(
        name='beta_a', shape=[1, num_classes]
      )

      # output poses and activations via matrix capsules_em_routing algorithm
      # poses: [N, O, PH x PW], activations: [N, O]
      poses, activations = self.matrix_capsules_em_routing(
        votes, i_activations, beta_v, beta_a, iterations, name='em_routing'
      )

      # pose: [N, O, PH, PW]
      # poses = tf.reshape(
      #   poses, [
      #     votes_shape[0], votes_shape[4], votes_shape[5], votes_shape[6]
      #   ]
      # )
      poses = tf.reshape(
        poses, [
          -1, votes_shape[4], votes_shape[5], votes_shape[6]
        ]
      )

      # add into GraphKeys.SUMMARIES
      tf.summary.histogram(
        'activations', activations
      )

    return poses, activations

  @staticmethod
  def matrix_capsules_em_routing(votes, i_activations, beta_v, beta_a,
                                 iterations, name):
    """The EM routing between input capsules (i) and output capsules (o).

    :param votes: [N, OH, OW, KH x KW x I, O, PH x PW] from capsule_conv(),
      or [N, KH x KW x I, O, PH x PW] from capsule_fc()
    :param i_activations: [N, OH, OW, KH x KW x I, O] from capsule_conv(),
      or [N, KH x KW x I, O] from capsule_fc()
    :param beta_v: [1, 1, 1, O] from capsule_conv(),
      or [1, O] from capsule_fc()
    :param beta_a: [1, 1, 1, O] from capsule_conv(),
      or [1, O] from capsule_fc()
    :param iterations: number of iterations in EM routing, often 3.
    :param name: name.

    :return: (pose, activation) of output capsules.

    note: the comment assumes arguments from capsule_conv(), remove OH, OW if from capsule_fc(),
      the function make sure is applicable to both cases by using negative index in argument axis.
    """

    # stop gradient in EM
    # votes = tf.stop_gradient(votes)

    # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
    votes_shape = votes.get_shape().as_list()
    # i_activations: [N, OH, OW, KH x KW x I]

    with tf.variable_scope(name):

      # note: match rr shape, i_activations shape with votes shape for broadcasting in EM routing

      # rr: [1, 1, 1, KH x KW x I, O, 1],
      # rr: routing matrix from each input capsule (i) to each output capsule (o)
      rr = tf.constant(
        1.0 / votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32
      )
      # rr = tf.Print(
      #   rr, [rr.shape, rr[0, ..., :, :, :]], 'rr', summarize=20
      # )

      # i_activations: expand_dims to [N, OH, OW, KH x KW x I, 1, 1]
      i_activations = i_activations[..., tf.newaxis, tf.newaxis]
      # i_activations = tf.Print(
      #   i_activations, [i_activations.shape, i_activations[0, ..., :, :, :]], 'i_activations', summarize=20
      # )

      # beta_v and beta_a: expand_dims to [1, 1, 1, 1, O, 1]
      beta_v = beta_v[..., tf.newaxis, :, tf.newaxis]
      beta_a = beta_a[..., tf.newaxis, :, tf.newaxis]

      def m_step(rr_, votes_, i_activations_, beta_v_,
                 beta_a_, inverse_temperature_):
        """The M-Step in EM Routing.

        :param rr_: [1, 1, 1, KH x KW x I, O, 1], or [N, KH x KW x I, O, 1],
          routing assignments from each input capsules (i) to each output capsules (o).
        :param votes_: [N, OH, OW, KH x KW x I, O, PH x PW], or [N, KH x KW x I, O, PH x PW],
          input capsules poses x view transformation.
        :param i_activations_: [N, OH, OW, KH x KW x I, 1, 1], or [N, KH x KW x I, 1, 1],
          input capsules activations, with dimensions expanded to match votes for broadcasting.
        :param beta_v_: cost of describing capsules with one variance in each h-th compenents,
          should be learned discriminatively.
        :param beta_a_: cost of describing capsules with one mean in across all h-th compenents,
          should be learned discriminatively.
        :param inverse_temperature_: lambda, increase over steps with respect to a fixed schedule.

        :return: (o_mean, o_stdv, o_activation)
        """

        # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
        # votes_shape = votes.get_shape().as_list()
        # votes = tf.Print(
        #   votes, [votes.shape, votes[0, ..., :, 0, :]], 'mstep: votes', summarize=20
        # )

        # rr_prime: [N, OH, OW, KH x KW x I, O, 1]
        rr_prime = rr_ * i_activations_
        # rr_prime = tf.Print(
        #   rr_prime, [rr_prime.shape, rr_prime[0, ..., :, 0, :]], 'mstep: rr_prime', summarize=20
        # )

        # rr_prime_sum: sum acorss i, [N, OH, OW, 1, O, 1]
        rr_prime_sum = tf.reduce_sum(
          rr_prime, axis=-3, keep_dims=True, name='rr_prime_sum'
        )
        # rr_prime_sum = tf.Print(
        #   rr_prime_sum, [rr_prime_sum.shape, rr_prime_sum[0, ..., :, 0, :]], 'mstep: rr_prime_sum', summarize=20
        # )

        # o_mean: [N, OH, OW, 1, O, PH x PW]
        o_mean_ = tf.reduce_sum(
          rr_prime * votes_, axis=-3, keep_dims=True
        ) / rr_prime_sum
        # o_mean = tf.Print(
        #   o_mean, [o_mean.shape, o_mean[0, ..., :, 0, :]], 'mstep: o_mean', summarize=20
        # )

        # o_stdv: [N, OH, OW, 1, O, PH x PW]
        o_stdv_ = tf.sqrt(
          tf.reduce_sum(
            rr_prime * tf.square(votes_ - o_mean_), axis=-3, keep_dims=True
          ) / rr_prime_sum
        )
        # o_stdv = tf.Print(
        #   o_stdv, [o_stdv.shape, o_stdv[0, ..., :, 0, :]], 'mstep: o_stdv', summarize=20
        # )

        # o_cost_h: [N, OH, OW, 1, O, PH x PW]
        o_cost_h = (beta_v_ + tf.log(o_stdv_ + epsilon)) * rr_prime_sum
        # o_cost_h = tf.Print(
        #   o_cost_h, [beta_v, o_cost_h.shape, o_cost[0, ..., :, 0, :]], 'mstep: beta_v, o_cost_h', summarize=20
        # )

        # o_activation: [N, OH, OW, 1, O, 1]
        # o_activations_cost = (beta_a - tf.reduce_sum(o_cost_h, axis=-1, keep_dims=True))
        # yg: This is to stable o_cost, which often large numbers, using an idea like batch norm.
        # It is in fact only the relative variance between each channel determined which one should activate,
        # the `relative` smaller variance, the `relative` higher activation.
        # o_cost: [N, OH, OW, 1, O, 1]
        o_cost = tf.reduce_sum(o_cost_h, axis=-1, keep_dims=True)
        o_cost_mean = tf.reduce_mean(o_cost, axis=-2, keep_dims=True)
        o_cost_stdv = tf.sqrt(
          tf.reduce_sum(
            tf.square(o_cost - o_cost_mean), axis=-2, keep_dims=True
          ) / o_cost.get_shape().as_list()[-2]
        )
        o_activations_cost = beta_a_ + (o_cost_mean - o_cost) / (
              o_cost_stdv + epsilon)

        # try to find a good inverse_temperature, for o_activation,
        # o_activations_cost = tf.Print(
        #   o_activations_cost, [
        #     beta_a[0, ..., :, :, :], inverse_temperature, o_activations_cost.shape, o_activations_cost[0, ..., :, :, :]
        #   ], 'mstep: beta_a, inverse_temperature, o_activation_cost', summarize=20
        # )
        tf.summary.histogram('o_activation_cost', o_activations_cost)
        o_activations_ = tf.sigmoid(
          inverse_temperature_ * o_activations_cost
        )
        # o_activations = tf.Print(
        #   o_activations, [o_activations.shape, o_activations[0, ..., :, :, :]], 'mstep: o_activation', summarize=20
        # )
        tf.summary.histogram('o_activation', o_activations_)

        return o_mean_, o_stdv_, o_activations_

      def e_step(o_mean_, o_stdv_, o_activations_, votes_):
        """The E-Step in EM Routing.

        :param o_mean_: [N, OH, OW, 1, O, PH x PW], or [N, 1, O, PH x PW],
        :param o_stdv_: [N, OH, OW, 1, O, PH x PW], or [N, 1, O, PH x PW],
        :param o_activations_: [N, OH, OW, 1, O, 1], or [N, 1, O, 1],
        :param votes_: [N, OH, OW, KH x KW x I, O, PH x PW], or [N, KH x KW x I, O, PH x PW],

        :return: rr
        """

        # votes: [N, OH, OW, KH x KW x I, O, PH x PW]
        # votes_shape = votes.get_shape().as_list()
        # votes = tf.Print(
        #   votes, [votes.shape, votes[0, ..., :, 0, :]], 'estep: votes', summarize=20
        # )

        # o_p: [N, OH, OW, KH x KW x I, O, 1]
        # o_p is the probability density of the h-th component of the vote from i to c
        o_p_unit0 = - tf.reduce_sum(
          tf.square(votes_ - o_mean_) / (2 * tf.square(o_stdv_)), axis=-1,
          keep_dims=True
        )
        # o_p_unit0 = tf.Print(
        #   o_p_unit0, [o_p_unit0.shape, o_p_unit0[0, ..., :, 0, :]], 'estep: o_p_unit0', summarize=20
        # )
        # o_p_unit1 = - tf.log(
        #   0.50 * votes_shape[-1] * tf.log(2 * pi) + epsilon
        # )
        # o_p_unit1 = tf.Print(
        #   o_p_unit1, [o_p_unit1.shape, o_p_unit1[0, ..., :, 0, :]], 'estep: o_p_unit1', summarize=20
        # )
        o_p_unit2 = - tf.reduce_sum(
          tf.log(o_stdv_ + epsilon), axis=-1, keep_dims=True
        )
        # o_p_unit2 = tf.Print(
        #   o_p_unit2, [o_p_unit2.shape, o_p_unit2[0, ..., :, 0, :]], 'estep: o_p_unit2', summarize=20
        # )
        # o_p
        o_p = o_p_unit0 + o_p_unit2
        # o_p = tf.Print(
        #   o_p, [o_p.shape, o_p[0, ..., :, 0, :]], 'estep: o_p', summarize=20
        # )
        # rr: [N, OH, OW, KH x KW x I, O, 1]
        # tf.nn.softmax() dim: either positive or -1?
        # https://github.com/tensorflow/tensorflow/issues/14916
        zz = tf.log(o_activations_ + epsilon) + o_p
        rr_ = tf.nn.softmax(
          zz, dim=len(zz.get_shape().as_list()) - 2
        )
        # rr = tf.Print(
        #   rr, [rr.shape, rr[0, ..., :, 0, :]], 'estep: rr', summarize=20
        # )
        tf.summary.histogram('rr', rr_)

        return rr_

      # inverse_temperature (min, max)
      # y=tf.sigmoid(x): y=0.50,0.73,0.88,0.95,0.98,0.99995458 for x=0,1,2,3,4,10,
      it_min = 1.0
      it_max = min(iterations, 3.0)
      o_mean, o_activations = None, None
      for it in range(iterations):
        inverse_temperature = it_min + (it_max - it_min) * it / max(1.0,
                                                                    iterations - 1.0)
        o_mean, o_stdv, o_activations = m_step(
          rr, votes, i_activations, beta_v, beta_a,
          inverse_temperature_=inverse_temperature
        )
        if it < iterations - 1:
          rr = e_step(
            o_mean, o_stdv, o_activations, votes
          )
          # stop gradient on m_step() output
          # https://www.tensorflow.org/api_docs/python/tf/stop_gradient
          # The EM algorithm where the M-step should not involve backpropagation through the output of the E-step.
          # rr = tf.stop_gradient(rr)

      # pose: [N, OH, OW, O, PH x PW] via squeeze o_mean [N, OH, OW, 1, O, PH x PW]
      poses = tf.squeeze(o_mean, axis=-3)

      # activation: [N, OH, OW, O] via squeeze o_activationis [N, OH, OW, 1, O, 1]
      activations = tf.squeeze(o_activations, axis=[-3, -1])

    return poses, activations


class Dense2CapsLayer(object):

  def __init__(self,
               cfg,
               identity_map=True,
               num_caps=None,
               act_fn='relu',
               vec_dim=8,
               batch_size=None,
               reshape_mode='GAP'):
    """Generate a Capsule layer densely.

    Args:
      cfg: configuration
      identity_map: use identity map or full-connected layer
      act_fn: activation function of full-connected layer, needed if
              identity_map is False
      num_caps: number of output capsules, needed if identity_map is False
      vec_dim: dimensions of vectors of capsule
      batch_size: number of samples per batch
      reshape_mode: 'FLATTEN' or 'GAP'
    """
    self.cfg = cfg
    self.identity_map = identity_map
    self.num_caps = num_caps
    self.act_fn = act_fn
    self.vec_dim = vec_dim
    self.batch_size = batch_size
    self.reshape_mode = reshape_mode
    self.tensor_shape = None

  @property
  def params(self):
    """Parameters of this layer."""
    return {
      'identity_map': self.identity_map,
      'num_caps': self.num_caps,
      'act_fn': self.act_fn,
      'vec_dim': self.vec_dim,
      'batch_size': self.batch_size,
      'reshape_mode': self.reshape_mode
    }

  def _fc_layer(self,
                x,
                out_dim=None,
                act_fn='relu',
                use_bias=True,
                idx=0):
    """Single full_connected layer

    Args:
      x: input tensor
      out_dim: hidden units of full_connected layer
      act_fn: activation function
      use_bias: use bias
      idx: index of layer

    Returns:
      output tensor of full_connected layer
    """
    with tf.variable_scope('fc_{}'.format(idx)):
      activation_fn = get_act_fn(act_fn)
      weights_initializer = tf.contrib.layers.xavier_initializer()

      if self.cfg.VAR_ON_CPU:
        weights = variable_on_cpu(
            name='weights',
            shape=[x.get_shape().as_list()[1], out_dim],
            initializer=weights_initializer,
            dtype=tf.float32)
        fc = tf.matmul(x, weights)

        if use_bias:
          biases = variable_on_cpu(
              name='biases',
              shape=[out_dim],
              initializer=tf.zeros_initializer(),
              dtype=tf.float32)
          fc = tf.add(fc, biases)

        if activation_fn is not None:
          fc = activation_fn(fc)

      else:
        biases_initializer = tf.zeros_initializer() if use_bias else None
        fc = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=out_dim,
            activation_fn=activation_fn,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer)

      return fc

  def __call__(self, inputs):
    """Convert inputs to capsule layer densely.

    Args:
      inputs: input tensor
        - shape: (batch_size, height, width, depth)

    Returns:
      tensor of capsules
        - shape: (batch_size, num_caps_j, vec_dim_j, 1)
    """
    with tf.variable_scope('dense2caps'):

      inputs_shape = inputs.get_shape().as_list()

      if len(inputs_shape) != 2:
        # Flatten shape: (batch_size, height * width * depth)
        if self.reshape_mode == 'FLATTEN':
          inputs_flatten = tf.contrib.layers.flatten(inputs)
        elif self.reshape_mode == 'GAP':
          inputs_flatten = tf.reduce_mean(inputs, axis=[1, 2])
        else:
          raise ValueError('Wrong reshape_mode!')
        assert inputs_flatten.get_shape() == (inputs_shape[0], inputs_shape[3])
      else:
        inputs_flatten = inputs

      if self.identity_map:
        self.num_caps = inputs_flatten.get_shape().as_list()[1]
        inputs_flatten = tf.expand_dims(inputs_flatten, -1)
        caps = tf.tile(inputs_flatten, [1, 1, self.vec_dim])
      else:
        caps_ = []
        for i in range(self.vec_dim):
          fc_ = self._fc_layer(x=inputs_flatten,
                               out_dim=self.num_caps,
                               act_fn=self.act_fn,
                               use_bias=True,
                               idx=i)
          fc_ = tf.expand_dims(fc_, -1)
          caps_.append(fc_)
        caps = tf.concat(caps_, axis=-1)

      # Reshape and generating a capsule layer
      caps = tf.reshape(caps, [self.batch_size, -1, self.vec_dim, 1])
      # caps shape: (batch_size, num_caps_j, vec_dim_j, 1)
      assert caps.get_shape() == (
        self.batch_size, self.num_caps, self.vec_dim, 1)

      # Applying activation function
      caps_activated = ActivationFunc.squash(
          caps, self.batch_size, self.cfg.EPSILON)
      # caps_activated shape: (batch_size, num_caps_j, vec_dim_j, 1)
      assert caps_activated.get_shape() == (
        self.batch_size, self.num_caps, self.vec_dim, 1)

      self.tensor_shape = caps_activated.get_shape().as_list()
      return caps_activated


class Code2CapsLayer(object):

  def __init__(self,
               cfg,
               vec_dim=8,
               batch_size=None,
               share_weights=False):
    """Generate a Capsule layer densely.

    Args:
      cfg: configuration
      vec_dim: dimensions of vectors of capsule
      batch_size: number of samples per batch
    """
    self.cfg = cfg
    self.vec_dim = vec_dim
    self.batch_size = batch_size
    self.tensor_shape = None
    self.share_weights = share_weights

  @property
  def params(self):
    """Parameters of this layer."""
    return {
      'vec_dim': self.vec_dim,
      'batch_size': self.batch_size,
      'share_weights': self.share_weights,
    }

  def __call__(self, inputs):
    """Convert inputs to capsule layer densely.

    Args:
      inputs: input tensor
        - shape: (batch_size, height, width, depth)

    Returns:
      tensor of capsules
        - shape: (batch_size, num_caps_j, vec_dim_j, 1)
    """
    with tf.variable_scope('code2caps'):

      inputs_shape = inputs.get_shape().as_list()

      if inputs_shape[-1] % self.vec_dim != 0:
        raise ValueError

      if len(inputs_shape) != 2:
        num_caps_j = inputs_shape[1] * inputs_shape[2] * \
                     (inputs_shape[3] // self.vec_dim)
      else:
        num_caps_j = inputs_shape[1] // self.vec_dim

      caps = tf.reshape(inputs, [self.batch_size, -1, self.vec_dim, 1])
      # caps shape: (batch_size, num_caps_j, vec_dim_j, 1)
      assert caps.get_shape() == (
        self.batch_size, num_caps_j, self.vec_dim, 1)

      # Applying activation function
      caps_activated = ActivationFunc.squash(
          caps, self.batch_size, self.cfg.EPSILON)
      # caps_activated shape: (batch_size, num_caps_j, vec_dim_j, 1)
      assert caps_activated.get_shape() == (
        self.batch_size, num_caps_j, self.vec_dim, 1)

      self.tensor_shape = caps_activated.get_shape().as_list()
      return caps_activated
