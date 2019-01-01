from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.layers import *
from models.capsule_layers import *


def classifier(inputs, cfg, batch_size=None, is_training=None):

  if cfg.DATABASE_NAME == 'radical':
    num_classes = cfg.NUM_RADICALS
  else:
    num_classes = 10

  model = Sequential(inputs)
  model.add(ConvLayer(
      cfg,
      kernel_size=9,
      stride=1,
      n_kernel=256,
      padding='VALID',
      act_fn='relu',
      idx=0
  ))
  model.add(Conv2CapsLayer(
      cfg,
      kernel_size=9,
      stride=2,
      n_kernel=32,
      vec_dim=8,
      padding='VALID',
      batch_size=batch_size
  ))
  model.add(CapsLayer(
      cfg,
      num_caps=10,
      vec_dim=8,
      route_epoch=3,
      batch_size=batch_size,
      idx=0
  ))
  model.add(CapsLayer(
      cfg,
      num_caps=num_classes,
      vec_dim=16,
      route_epoch=3,
      batch_size=batch_size,
      idx=1
  ))

  return model.top_layer, model.info


def decoder(inputs, cfg, batch_size=None, is_training=None):

  model = Sequential(inputs)
  act_fn_last = None if cfg.REC_LOSS == 'ce' else 'relu'

  if (cfg.DATABASE_NAME == 'radical') or (cfg.DATABASE_NAME == 'cifar10'):
      model.add(DenseLayer(
          cfg,
          out_dim=512,
          act_fn='relu',
          idx=0))
      model.add(DenseLayer(
          cfg,
          out_dim=1024,
          act_fn='relu',
          idx=1))
      model.add(DenseLayer(
          cfg,
          out_dim=cfg.IMAGE_SIZE[0]*cfg.IMAGE_SIZE[1],
          act_fn=act_fn_last,
          idx=2))

  elif cfg.DATABASE_NAME == 'mnist':
      model.add(DenseLayer(
          cfg,
          out_dim=512,
          act_fn='relu',
          idx=0))
      model.add(DenseLayer(
          cfg,
          out_dim=1024,
          act_fn='relu',
          idx=1))
      model.add(DenseLayer(
          cfg,
          out_dim=cfg.IMAGE_SIZE[0]*cfg.IMAGE_SIZE[1],
          act_fn=act_fn_last,
          idx=2))

  else:
    raise ValueError('Wrong database name!')

  return model.top_layer, model.info


basel_arch = {
  'classifier': classifier,
  'decoder': decoder
}
