import pickle
import numpy as np
from PIL import Image
from models import utils
from tqdm import tqdm

import os
import tensorflow as tf
from keras import backend as K


class GetBottleneckFeatures(object):
  """Save bottleneck features of transfer learning models."""

  def __init__(self, model_name):
    self.model_name = model_name
    # Only shows Errors
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

  def _extract_features(self,
                        tensor,
                        weights='imagenet',
                        include_top=False,
                        pooling=None):
    """Extract bottleneck features from transfer learning models."""
    if self.model_name == 'vgg16':
      from keras.applications.vgg16 import VGG16, preprocess_input
      return VGG16(weights=weights,
                   include_top=include_top,
                   pooling=pooling).predict(preprocess_input(tensor))
    elif self.model_name == 'vgg19':
      from keras.applications.vgg19 import VGG19, preprocess_input
      return VGG19(weights=weights,
                   include_top=include_top,
                   pooling=pooling).predict(preprocess_input(tensor))
    elif self.model_name == 'resnet50':
      from keras.applications.resnet50 import ResNet50, preprocess_input
      return ResNet50(weights=weights,
                      include_top=include_top,
                      pooling=pooling).predict(preprocess_input(tensor))
    elif self.model_name == 'inceptionv3':
      from keras.applications.inception_v3 import \
        InceptionV3, preprocess_input
      return InceptionV3(weights=weights,
                         include_top=include_top,
                         pooling=pooling).predict(preprocess_input(tensor))
    elif self.model_name == 'xception':
      from keras.applications.xception import Xception, preprocess_input
      return Xception(weights=weights,
                      include_top=include_top,
                      pooling=pooling).predict(preprocess_input(tensor))
    else:
      raise ValueError('Wrong transfer learning model name!')

  def _get_bottleneck_feature_shape(self, pooling=None):
    """Get bottleneck feature shapes."""
    if self.model_name == 'vgg16':
      bf_shape = (512,) if pooling is not None else (7, 7, 512)
    elif self.model_name == 'vgg19':
      bf_shape = (512,) if pooling is not None else (7, 7, 512)
    elif self.model_name == 'resnet50':
      bf_shape = (2048,) if pooling is not None else (1, 1, 2048)
    elif self.model_name == 'inceptionv3':
      bf_shape = (2048,) if pooling is not None else (5, 5, 2048)
    elif self.model_name == 'xception':
      bf_shape = (2048,) if pooling is not None else (7, 7, 2048)
    else:
      raise ValueError('Wrong transfer learning model name!')

    return bf_shape

  def get_bottleneck_features(self,
                              inputs,
                              batch_size=None,
                              data_type=np.float32,
                              pooling='avg'):
    # Check image size for transfer learning models
    inputs_shape = inputs.shape
    assert inputs_shape[1:3] == (224, 224)

    # Scale to 0-255 and extract features
    if batch_size:
      batch_generator = utils.get_batches(
          inputs, batch_size=batch_size, keep_last=True)
      n_batch = len(inputs) // batch_size + 1
      bottleneck_features = []
      for _ in tqdm(range(n_batch), total=n_batch, ncols=100, unit='batch'):

        inputs_batch = next(batch_generator)
        inputs_batch = utils.imgs_scale_to_255(inputs_batch).astype(data_type)

        if inputs_shape[3] == 1:
          inputs_batch = np.concatenate(
              [inputs_batch, inputs_batch, inputs_batch], axis=-1)

        assert inputs_batch.shape[1:] == (224, 224, 3)
        bf_batch = self._extract_features(inputs_batch, pooling=pooling)
        bottleneck_features.append(bf_batch)

        # Release memory
        K.clear_session()
        tf.reset_default_graph()

      bottleneck_features = np.concatenate(bottleneck_features, axis=0)
    else:
      inputs = utils.imgs_scale_to_255(inputs).astype(data_type)
      if inputs_shape[3] == 1:
        inputs = np.concatenate([inputs, inputs, inputs], axis=-1)
      bottleneck_features = self._extract_features(inputs, pooling=pooling)

    # Check data shape
    assert len(bottleneck_features) == len(inputs)
    assert bottleneck_features.shape[1:] == \
        self._get_bottleneck_feature_shape(pooling=pooling)

    return bottleneck_features

  def save_bottleneck_features(self,
                               inputs,
                               file_path,
                               batch_size=None,
                               pooling='avg',
                               data_type=np.float32):
    inputs_shape = inputs.shape
    img_mode = 'L' if inputs_shape[3] == 1 else 'RGB'

    with open(file_path, 'wb') as f:

      if batch_size:

        batch_generator = utils.get_batches(
            inputs, batch_size=batch_size, keep_last=True)
        if len(inputs) % batch_size == 0:
          n_batch = len(inputs) // batch_size
        else:
          n_batch = len(inputs) // batch_size + 1

        for _ in tqdm(range(n_batch), total=n_batch, ncols=100, unit='batch'):
          inputs_batch = next(batch_generator)
          inputs_batch = utils.imgs_scale_to_255(inputs_batch).astype(data_type)

          if inputs_shape[1:3] != (224, 224):
            inputs_batch = utils.img_resize(inputs_batch,
                                            (224, 224),
                                            img_mode=img_mode,
                                            resize_filter=Image.ANTIALIAS,
                                            verbose=False,
                                            ).astype(data_type)
          if inputs_shape[3] == 1:
            inputs_batch = np.concatenate(
                [inputs_batch, inputs_batch, inputs_batch], axis=-1)

          assert inputs_batch.shape[1:] == (224, 224, 3)
          bf_batch = self._extract_features(inputs_batch, pooling=pooling)
          assert bf_batch.shape[1:] == \
              self._get_bottleneck_feature_shape(pooling=pooling)
          f.write(pickle.dumps(bf_batch))

          # Release memory
          K.clear_session()
          tf.reset_default_graph()

      else:
        inputs = utils.imgs_scale_to_255(inputs).astype(data_type)

        if inputs_shape[3] == 1:
          inputs = np.concatenate([inputs, inputs, inputs], axis=-1)

        assert inputs.shape[1:] == (224, 224, 3)
        bottleneck_features = self._extract_features(inputs, pooling=pooling)
        assert bottleneck_features.shape[1:] == \
            self._get_bottleneck_feature_shape(pooling=pooling)
        f.write(pickle.dumps(bottleneck_features))
