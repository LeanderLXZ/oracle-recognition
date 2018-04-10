from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def variable_on_cpu(name,
                    shape,
                    initializer,
                    dtype=tf.float32,
                    trainable=True):
  """
  Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    dtype: data type
    trainable: variable can be trained by models
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer,
                          dtype=dtype, trainable=trainable)
  return var


def get_act_fn(act_fn):
  """
  Helper to get activation function from name.
  """
  if act_fn == 'relu':
    activation_fn = tf.nn.relu
  elif act_fn == 'sigmoid':
    activation_fn = tf.nn.sigmoid
  elif act_fn == 'elu':
    activation_fn = tf.nn.elu
  elif act_fn is None:
    activation_fn = None
  else:
    raise ValueError('Wrong activation function name!')
  return activation_fn


class ModelBase(object):

  def __init__(self, cfg):

    self.cfg = cfg

  @staticmethod
  def _avg_pool(x,
                pool_size=None,
                stride=None,
                padding='SAME'):
    """
    Average pooling
    """
    with tf.name_scope('avg_pool'):
      return tf.layers.average_pooling2d(
          inputs=x,
          pool_size=pool_size,
          strides=stride,
          padding=padding)

  @staticmethod
  def _global_avg_pool(x):
    """
    Average pooling on full image
    """
    with tf.name_scope('global_avg_pool'):
      assert x.get_shape().ndims == 4
      return tf.reduce_mean(x, [1, 2])


class DenseLayer(object):

  def __init__(self,
               cfg,
               out_dim=None,
               act_fn='relu',
               use_bias=True,
               idx=0):
    """
    Single convolution layer

    Args:
      out_dim: hidden units of full_connected layer
      act_fn: activation function
      use_bias: use bias
      idx: index of layer
    """
    self.cfg = cfg
    self.out_dim = out_dim
    self.act_fn = act_fn
    self.use_bias = use_bias
    self.idx = idx

  @property
  def params(self):
    """
    Parameters of this layer.
    """
    return {'out_dim': self.out_dim,
            'act_fn': self.act_fn,
            'use_bias': self.use_bias,
            'idx': self.idx}

  def __call__(self, inputs):
    """
    Single full-connected layer

    Args:
      inputs: input tensor
        - shape: (batch_size, num_units)
    Returns:
      output tensor of full-connected layer
    """
    with tf.variable_scope('fc_{}'.format(self.idx)):
      activation_fn = get_act_fn(self.act_fn)
      weights_initializer = tf.contrib.layers.xavier_initializer()

      if self.cfg.VAR_ON_CPU:
        weights = variable_on_cpu(
            name='weights',
            shape=[inputs.get_shape().as_list()[1], self.out_dim],
            initializer=weights_initializer,
            dtype=tf.float32)
        fc = tf.matmul(inputs, weights)

        if self.use_bias:
          biases = variable_on_cpu(
              name='biases',
              shape=[self.out_dim],
              initializer=tf.zeros_initializer(),
              dtype=tf.float32)
          fc = tf.add(fc, biases)

        if activation_fn is not None:
          fc = activation_fn(fc)

      else:
        biases_initializer = tf.zeros_initializer() if self.use_bias else None
        fc = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=self.out_dim,
            activation_fn=activation_fn,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer)

      return fc


class ConvLayer(object):

  def __init__(self,
               cfg,
               kernel_size=None,
               stride=None,
               n_kernel=None,
               padding='SAME',
               act_fn='relu',
               w_init_fn=tf.contrib.layers.xavier_initializer(),
               resize=None,
               use_bias=True,
               atrous=False,
               idx=0):
    """
    Single convolution layer

    Args:
      cfg: configuration
      kernel_size: size of convolution kernel
      stride: stride of convolution kernel
      n_kernel: number of convolution kernels
      padding: padding type of convolution kernel
      act_fn: activation function
      w_init_fn: weights initializer of convolution layer
      resize: if resize is not None, resize every image
      atrous: use atrous convolution
      use_bias: use bias
      idx: index of layer
    """
    self.cfg = cfg
    self.kernel_size = kernel_size
    self.stride = stride
    self.n_kernel = n_kernel
    self.padding = padding
    self.act_fn = act_fn
    self.w_init_fn = w_init_fn
    self.resize = resize
    self.use_bias = use_bias
    self.atrous = atrous
    self.idx = idx

  @property
  def params(self):
    """Parameters of this layer."""
    return {'kernel_size': self.kernel_size,
            'stride': self.stride,
            'n_kernel': self.n_kernel,
            'padding': self.padding,
            'act_fn': self.act_fn,
            'w_init_fn': self.w_init_fn,
            'resize': self.resize,
            'use_bias': self.use_bias,
            'atrous': self.atrous,
            'idx': self.idx}

  def __call__(self, inputs):
    """
    Single convolution layer

    Args:
      inputs: input tensor
        - shape: (batch_size, height, width, channel)
    Returns:
      output tensor of convolution layer
    """
    with tf.variable_scope('conv_{}'.format(self.idx)):
      # Resize image
      if self.resize is not None:
        inputs = tf.image.resize_nearest_neighbor(
            inputs, (self.resize, self.resize))

      # With atrous
      if not self.atrous and self.stride > 1:
        pad = self.kernel_size - 1
        pad_beg = pad // 2
        pad_end = pad - pad_beg
        inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        self.padding = 'VALID'

      activation_fn = get_act_fn(self.act_fn)

      if self.cfg.VAR_ON_CPU:
        kernels = variable_on_cpu(
            name='kernels',
            shape=[self.kernel_size, self.kernel_size,
                   inputs.get_shape().as_list()[3], self.n_kernel],
            initializer=self.w_init_fn,
            dtype=tf.float32)
        conv = tf.nn.conv2d(input=inputs,
                            filter=kernels,
                            strides=[1, self.stride, self.stride, 1],
                            padding=self.padding)

        if self.use_bias:
          biases = variable_on_cpu(
              name='biases',
              shape=[self.n_kernel],
              initializer=tf.zeros_initializer(),
              dtype=tf.float32)
          conv = tf.nn.bias_add(conv, biases)

        if activation_fn is not None:
          conv = activation_fn(conv)

      else:
        biases_initializer = tf.zeros_initializer() if self.use_bias else None
        conv = tf.contrib.layers.conv2d(
            inputs=inputs,
            num_outputs=self.n_kernel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            activation_fn=activation_fn,
            weights_initializer=self.w_init_fn,
            biases_initializer=biases_initializer)

      return conv


class ConvTLayer(object):

  def __init__(self,
               cfg,
               kernel_size=None,
               stride=None,
               n_kernel=None,
               padding='SAME',
               act_fn='relu',
               output_shape=None,
               w_init_fn=tf.contrib.layers.xavier_initializer(),
               use_bias=True,
               idx=None):
    """
    Single transpose convolution layer

    Args:
      cfg: configuration
      kernel_size: size of convolution kernel
      stride: stride of convolution kernel
      n_kernel: number of convolution kernels
      padding: padding type of convolution kernel
      act_fn: activation function
      output_shape: output shape of deconvolution layer
      w_init_fn: weights initializer of convolution layer
      use_bias: use bias
      idx: index of layer
    """
    self.cfg = cfg
    self.kernel_size = kernel_size
    self.stride = stride
    self.n_kernel = n_kernel
    self.padding = padding
    self.act_fn = act_fn
    self.output_shape = output_shape
    self.w_init_fn = w_init_fn
    self.use_bias = use_bias
    self.idx = idx

  @property
  def params(self):
    """Parameters of this layer."""
    return {'kernel_size': self.kernel_size,
            'stride': self.stride,
            'n_kernel': self.n_kernel,
            'padding': self.padding,
            'act_fn': self.act_fn,
            'output_shape': self.output_shape,
            'w_init_fn': self.w_init_fn,
            'use_bias': self.use_bias,
            'idx': self.idx}

  def __call__(self, inputs):
    """
    Single transpose convolution layer

    Args:
      inputs: input tensor
        - shape: (batch_size, height, width, channel)
    Returns:
      output tensor of transpose convolution layer
    """
    with tf.variable_scope('conv_t_{}'.format(self.idx)):
      activation_fn = get_act_fn(self.act_fn)

      if self.cfg.VAR_ON_CPU:
        kernels = variable_on_cpu(
            name='kernels',
            shape=[self.kernel_size, self.kernel_size,
                   self.n_kernel, inputs.get_shape().as_list()[3]],
            initializer=self.w_init_fn,
            dtype=tf.float32)
        conv_t = tf.nn.conv2d_transpose(
            value=inputs,
            filter=kernels,
            output_shape=self.output_shape,
            strides=[1, self.stride, self.stride, 1],
            padding=self.padding)

        if self.use_bias:
          biases = variable_on_cpu(
              name='biases',
              shape=[self.n_kernel],
              initializer=tf.zeros_initializer(),
              dtype=tf.float32)
          conv_t = tf.nn.bias_add(conv_t, biases)

        if activation_fn is not None:
          conv_t = activation_fn(conv_t)

      else:
        biases_initializer = tf.zeros_initializer() if self.use_bias else None
        conv_t = tf.contrib.layers.conv2d_transpose(
            inputs=inputs,
            num_outputs=self.n_kernel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            activation_fn=activation_fn,
            weights_initializer=self.w_init_fn,
            biases_initializer=biases_initializer)

      return conv_t


class BatchNorm(object):

  def __init__(self,
               cfg,
               is_training,
               momentum=0.99,
               center=True,
               scale=True,
               epsilon=0.001,
               act_fn='relu',
               idx=None):
    """
    Batch normalization layer.

    Args:
      cfg: configuration
      is_training: Whether or not the layer is in training mode.
      momentum: Momentum for the moving average.
      center: If True, add offset of beta to normalized tensor.
              If False, beta is ignored.
      scale: If True, multiply by gamma. If False, gamma is not used.
      epsilon: Small float added to variance to avoid dividing by zero.
      act_fn: Add a activation function after batch normalization layer.
              If None, not add.
      idx: index of layer
    """
    self.cfg = cfg
    self.is_training = is_training
    self.momentum = momentum
    self.center = center
    self.scale = scale
    self.epsilon = epsilon
    self.act_fn = act_fn
    self.idx = idx

  @property
  def params(self):
    """Parameters of this layer."""
    return {'cfg': self.cfg,
            'momentum': self.momentum,
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'act_fn': self.act_fn}

  def __call__(self, inputs):
    """
    Batch normalization layer.

    Args:
      inputs: input tensor
    Returns:
      reshaped tensor
    """
    with tf.variable_scope('batch_norm_{}'.format(self.idx)):
      bn = tf.layers.batch_normalization(
          inputs=inputs,
          momentum=self.momentum,
          center=self.center,
          scale=self.scale,
          epsilon=self.epsilon,
          training=self.is_training)

      if self.act_fn is not None:
        activation_fn = get_act_fn(self.act_fn)
        return activation_fn(bn)
      else:
        return bn


class Reshape(object):

  def __init__(self, shape, name=None):
    """
    Reshape a tensor.

    Args:
      shape:shape of output tensor
      name: name of output tensor
    """
    self.shape = shape
    self.name = name

  @property
  def params(self):
    """Parameters of this layer."""
    return {'shape': self.shape,
            'name': self.name}

  def __call__(self, inputs):
    """
    Reshape a tensor.

    Args:
      inputs: input tensor
    Returns:
      reshaped tensor
    """
    return tf.reshape(inputs, shape=self.shape, name=self.name)


class Sequential(object):
  """
  Build models architecture by sequential.
  """
  def __init__(self, inputs):
    self._top = inputs
    self._info = []

  def add(self, layer):
    """
    Add a layer to the top of the models.

    Args:
      layer: the layer to be added
    """
    self._top = layer(self._top)
    layer_name_ = layer.__class__.__name__
    layer_params_ = layer.params
    self._info.append((layer_name_, layer_params_))

  @property
  def top_layer(self):
    """The top layer of the models."""
    return self._top

  @property
  def info(self):
    """The architecture information of the models."""
    return self._info
