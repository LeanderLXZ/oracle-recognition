from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.model_base import *
from models.capsule_layer import *


def classifier(inputs, cfg, batch_size=None, is_training=None):

  model = Sequential(inputs)
  model.add(ConvLayer(
      cfg,
      kernel_size=5,
      stride=1,
      n_kernel=64,
      padding='VALID',
      act_fn=None,
      idx=0
  ))
  model.add(BatchNorm(
      cfg, is_training, momentum=0.99, act_fn='relu', idx=0))
  model.add(ConvLayer(
      cfg,
      kernel_size=5,
      stride=1,
      n_kernel=128,
      padding='VALID',
      act_fn=None,
      idx=1
  ))
  model.add(BatchNorm(
      cfg, is_training, momentum=0.99, act_fn='relu', idx=1))
  # model.add(ConvLayer(
  #     cfg,
  #     kernel_size=5,
  #     stride=1,
  #     n_kernel=256,
  #     padding='VALID',
  #     act_fn=None,
  #     idx=2
  # ))
  # model.add(BatchNorm(
  #     cfg, is_training, momentum=0.99, act_fn='relu', idx=2))
  # models.add(Dense2Capsule(
  #     cfg,
  #     identity_map=True,
  #     num_caps=None,
  #     act_fn='relu',
  #     vec_dim=8,
  #     batch_size=batch_size
  # ))
  model.add(Conv2CapsLayer(
      cfg,
      kernel_size=5,
      stride=2,
      n_kernel=16,
      vec_dim=16,
      padding='VALID',
      batch_size=batch_size
  ))
  # model.add(CapsLayer(
  #     cfg,
  #     num_caps=256,
  #     vec_dim=16,
  #     route_epoch=3,
  #     batch_size=batch_size,
  #     idx=0
  # ))
  model.add(CapsLayer(
      cfg,
      num_caps=148,
      vec_dim=32,
      route_epoch=3,
      batch_size=batch_size,
      idx=1
  ))

  return model.top_layer, model.info


def decoder(inputs, cfg, batch_size=None, is_training=None):

  model = Sequential(inputs)
  act_fn_last = None if cfg.RECONSTRUCTION_LOSS == 'ce' else 'relu'

  if cfg.DATABASE_NAME == 'mnist':
    if cfg.DECODER_TYPE == 'fc':
      model.add(DenseLayer(
          cfg,
          out_dim=512,
          idx=0))
      model.add(DenseLayer(
          cfg,
          out_dim=1024,
          idx=1))
      model.add(DenseLayer(
          cfg,
          out_dim=784,
          act_fn=act_fn_last,
          idx=2))

    elif cfg.DECODER_TYPE == 'conv':
      model.add(Reshape(      # (b, 4, 4, 1)
          (batch_size, 4, 4, -1), name='reshape'))
      model.add(ConvLayer(    # (b, 7, 7, 16)
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=16,
          resize=7,
          idx=0))
      model.add(ConvLayer(    # (b, 14, 14, 32)
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=32,
          resize=14,
          idx=1))
      model.add(ConvLayer(    # (b, 28, 28, 16)
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=16,
          resize=28,
          idx=2))
      model.add(ConvLayer(    # (b, 28, 28, 1)
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=1,
          act_fn=act_fn_last,
          idx=3))

    elif cfg.DECODER_TYPE == 'conv_t':
      model.add(Reshape(
          (batch_size, 1, 1, -1), name='reshape'))
      model.add(ConvTLayer(
          cfg,
          kernel_size=4,
          stride=1,
          n_kernel=16,
          output_shape=[batch_size, 4, 4, 16],
          padding='VALID',
          act_fn=None,
          idx=0))
      model.add(BatchNorm(
          cfg, is_training, momentum=0.99, act_fn='relu', idx=0))
      model.add(ConvTLayer(
          cfg,
          kernel_size=9,
          stride=1,
          n_kernel=32,
          output_shape=[batch_size, 12, 12, 32],
          padding='VALID',
          act_fn=None,
          idx=1))
      model.add(BatchNorm(
          cfg, is_training, momentum=0.99, act_fn='relu', idx=1))
      model.add(ConvTLayer(
          cfg,
          kernel_size=9,
          stride=1,
          n_kernel=16,
          output_shape=[batch_size, 20, 20, 16],
          padding='VALID',
          act_fn=None,
          idx=2))
      model.add(BatchNorm(
          cfg, is_training, momentum=0.99, act_fn='relu', idx=2))
      model.add(ConvTLayer(
          cfg,
          kernel_size=9,
          stride=1,
          n_kernel=8,
          output_shape=[batch_size, 28, 28, 8],
          padding='VALID',
          act_fn=None,
          idx=3))
      model.add(BatchNorm(
          cfg, is_training, momentum=0.99, act_fn='relu', idx=3))
      model.add(ConvTLayer(
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=1,
          output_shape=[batch_size, 28, 28, 1],
          act_fn=act_fn_last,
          idx=4))

    else:
      raise ValueError('Wrong decoder type!')

  elif cfg.DATABASE_NAME == 'cifar10':
    if cfg.DECODER_TYPE == 'fc':
      model.add(DenseLayer(
          cfg,
          out_dim=2048,
          idx=0))
      model.add(DenseLayer(
          cfg,
          out_dim=4096,
          idx=1))
      model.add(DenseLayer(
          cfg,
          out_dim=3072,
          act_fn=act_fn_last,
          idx=2))

    elif cfg.DECODER_TYPE == 'conv':
      model.add(Reshape(      # (b, 4, 4, 1)
          (batch_size, 4, 4, -1), name='reshape'))
      model.add(ConvLayer(    # (b, 8, 8, 16)
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=16,
          idx=0))
      model.add(ConvLayer(    # (b, 16, 16, 32)
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=32,
          resize=16,
          idx=1))
      model.add(ConvLayer(    # (b, 32, 32, 16)
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=16,
          resize=32,
          idx=2))
      model.add(ConvLayer(    # (b, 32, 32, 3)
          cfg,
          kernel_size=3,
          stride=1,
          n_kernel=3,
          act_fn=act_fn_last,
          idx=3))

    elif cfg.DECODER_TYPE == 'conv_t':
      model.add(Reshape(
          (batch_size, 1, 1, -1), name='reshape'))
      model.add(ConvTLayer(
          cfg,
          kernel_size=4,
          stride=1,
          n_kernel=16,
          output_shape=[batch_size, 4, 4, 16],
          padding='VALID',
          act_fn=None,
          idx=0))
      model.add(BatchNorm(
          cfg, is_training, momentum=0.99, act_fn='relu', idx=0))
      model.add(ConvTLayer(
          cfg,
          kernel_size=9,
          stride=1,
          n_kernel=32,
          output_shape=[batch_size, 12, 12, 32],
          padding='VALID',
          act_fn=None,
          idx=1))
      model.add(BatchNorm(
          cfg, is_training, momentum=0.99, act_fn='relu', idx=1))
      model.add(ConvTLayer(
          cfg,
          kernel_size=9,
          stride=1,
          n_kernel=16,
          output_shape=[batch_size, 20, 20, 16],
          padding='VALID',
          act_fn=None,
          idx=2))
      model.add(BatchNorm(
          cfg, is_training, momentum=0.99, act_fn='relu', idx=2))
      model.add(ConvTLayer(
          cfg,
          kernel_size=9,
          stride=1,
          n_kernel=8,
          output_shape=[batch_size, 28, 28, 8],
          padding='VALID',
          act_fn=None,
          idx=3))
      model.add(BatchNorm(
          cfg, is_training, momentum=0.99, act_fn='relu', idx=3))
      model.add(ConvTLayer(
          cfg,
          kernel_size=5,
          stride=1,
          n_kernel=3,
          output_shape=[batch_size, 32, 32, 3],
          padding='VALID',
          act_fn=act_fn_last,
          idx=4))

    else:
      raise ValueError('Wrong decoder type!')

  return model.top_layer, model.info
