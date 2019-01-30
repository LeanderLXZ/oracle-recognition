from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

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

  @property
  def params(self):
    """Parameters of this layer."""
    return {
      'num_caps': self.num_caps,
      'vec_dim': self.vec_dim,
      'route_epoch': self.route_epoch,
      'batch_size': self.batch_size,
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
