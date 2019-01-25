import numpy as np
from models import utils
from tqdm import tqdm


class GetBottleneckFeatures(object):
  """Save bottleneck features of transfer learning models."""

  def __init__(self, model_name):
    self.model_name = model_name

  def _extract_features(self, tensor):
    """Extract bottleneck features from transfer learning models."""
    if self.model_name == 'vgg16':
      from keras.applications.vgg16 import VGG16, preprocess_input
      return VGG16(weights='imagenet', include_top=False).predict(
          preprocess_input(tensor))
    elif self.model_name == 'vgg19':
      from keras.applications.vgg19 import VGG19, preprocess_input
      return VGG19(weights='imagenet', include_top=False).predict(
          preprocess_input(tensor))
    elif self.model_name == 'resnet50':
      from keras.applications.resnet50 import ResNet50, preprocess_input
      return ResNet50(weights='imagenet', include_top=False).predict(
          preprocess_input(tensor))
    elif self.model_name == 'inceptionv3':
      from keras.applications.inception_v3 import \
        InceptionV3, preprocess_input
      return InceptionV3(weights='imagenet', include_top=False).predict(
          preprocess_input(tensor))
    elif self.model_name == 'xception':
      from keras.applications.xception import Xception, preprocess_input
      return Xception(weights='imagenet', include_top=False).predict(
          preprocess_input(tensor))
    else:
      raise ValueError('Wrong transfer learning model name!')

  def _get_bottleneck_feature_shape(self):
    """Get bottleneck feature shapes."""
    if self.model_name == 'vgg16':
      bf_shape = (7, 7, 512)
    elif self.model_name == 'vgg19':
      bf_shape = (7, 7, 512)
    elif self.model_name == 'resnet50':
      bf_shape = (1, 1, 2048)
    elif self.model_name == 'inceptionv3':
      bf_shape = (5, 5, 2048)
    elif self.model_name == 'xception':
      bf_shape = (7, 7, 2048)
    else:
      raise ValueError('Wrong transfer learning model name!')

    return bf_shape

  def get_features(self, inputs, batch_zie=None):

    # Check image size for transfer learning models
    assert inputs.shape[1:3] == (224, 224)

    # Get bottleneck features
    utils.thin_line()
    print('Calculating bottleneck features...')

    # Scale to 0-255 and extract features
    if batch_zie:
      batch_generator = utils.get_batches_all_x(inputs, batch_zie)
      n_batch = len(inputs) // batch_zie + 1
      bottleneck_features = []
      for _ in tqdm(range(n_batch), total=n_batch, ncols=100, unit='batches'):
        inputs_batch = next(batch_generator)
        bf_batch = self._extract_features(
          utils.imgs_scale_to_255(inputs_batch))
        bottleneck_features.append(bf_batch)
      bottleneck_features = np.concatenate(bottleneck_features, axis=0)
    else:
      bottleneck_features = self._extract_features(
          utils.imgs_scale_to_255(inputs))

    # Check data shape
    assert len(bottleneck_features) == len(inputs)
    assert bottleneck_features.shape[1:] == self._get_bottleneck_feature_shape()

    return bottleneck_features
