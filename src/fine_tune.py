import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from models import utils
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

from config import config


def extract_vgg16():
  from keras.applications.vgg16 import VGG16, preprocess_input
  return VGG16(weights='imagenet', include_top=False), preprocess_input

def extract_vgg19():
  from keras.applications.vgg19 import VGG19, preprocess_input
  return VGG19(weights='imagenet', include_top=False), preprocess_input

def extract_resnet50():
  from keras.applications.resnet50 import ResNet50, preprocess_input
  return ResNet50(weights='imagenet', include_top=False), preprocess_input

def extract_xception():
  from keras.applications.xception import Xception, preprocess_input
  return Xception(weights='imagenet', include_top=False), preprocess_input

def extract_inceptionv3():
  from keras.applications.inception_v3 import InceptionV3, preprocess_input
  return InceptionV3(weights='imagenet', include_top=False), preprocess_input


class FineTune(object):

  def __init__(self, cfg, n_output, base_model_name, n_use_layers,
               n_freeze_layers=None, load_pre_model=False):
    self.cfg = cfg
    self.n_output = n_output
    self.base_model_name = base_model_name
    self.n_use_layers = n_use_layers
    self.n_freeze_layers = n_freeze_layers
    self.base_model, self.preprocess_input = self._get_base_model()

    if self.cfg.DATABASE_MODE is not None:
      self.preprocessed_path = join(
          '../data/{}'.format(self.cfg.DATABASE_MODE), self.cfg.DATABASE_NAME)
    else:
      self.preprocessed_path = join(
        self.cfg.DPP_DATA_PATH, self.cfg.DATABASE_NAME)

    self.x_train, self.y_train, self.x_valid, self.y_valid = \
      self._load_bottleneck_features()

    if load_pre_model:
      self.model = self._load_model()
      self.bf_model = \
        Model(input=self.model.input, output=self.model.layers[:-1].output)
    else:
      self.model = None
      self.bf_model = None

  def _load_bottleneck_features(self):
      """Load preprocessed bottleneck features."""
      utils.thick_line()
      print('Loading data...')
      utils.thin_line()

      x_train = utils.load_pkls(
        self.preprocessed_path, 'x_train')
      x_valid = utils.load_pkls(
        self.preprocessed_path, 'x_valid', add_n_batch=1)

      y_train = utils.load_pkls(self.preprocessed_path, 'y_train')
      y_valid = utils.load_pkls(self.preprocessed_path, 'y_valid')

      utils.thin_line()
      print('Data info:')
      utils.thin_line()
      print('x_train: {}\ny_train: {}\nx_valid: {}\ny_valid: {}'.format(
        x_train.shape,
        y_train.shape,
        x_valid.shape,
        y_valid.shape))

      return x_train, y_train, x_valid, y_valid

  @staticmethod
  def _save_model(model):
    model.save('saved_models/model-fine-tuned.h5')

  @staticmethod
  def _load_model():
    return load_model('saved_models/model-fine-tuned.h5')

  def _get_base_model(self):
    if self.base_model_name == 'vgg16':
      return extract_vgg16()
    elif self.base_model_name == 'vgg19':
      return extract_vgg19()
    elif self.base_model_name == 'resnet50':
      return extract_resnet50()
    elif self.base_model_name == 'inceptionv3':
      return extract_inceptionv3()
    elif self.base_model_name == 'xception':
      return extract_inceptionv3()

  def _model_arch(self, base_model):

    # Freeze the bottom layers and retrain the remaining top layers.
    if self.n_freeze_layers:
      for layer in base_model.layers[:self.n_freeze_layers]:
        layer.trainable = False
      for layer in base_model.layers[self.n_freeze_layers:]:
        layer.trainable = True
    else:
      for layer in base_model.layers:
        layer.trainable = False

    # Add last layer to the conv
    if self.n_use_layers:
      x = base_model.layes[self.n_use_layers].output
    else:
      x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(self.n_output, activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model

  def train(self, epochs=100, batch_size=32):

    self.model = self._model_arch(self.base_model)

    train_data_gen = ImageDataGenerator(
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
    train_generator = train_data_gen.flow(self.x_train, self.y_train,
                                         batch_size=batch_size)

    valid_data_gen = ImageDataGenerator()
    validation_generator = valid_data_gen.flow(self.x_valid, self.y_valid,
                                            batch_size=batch_size)

    checkpointer = ModelCheckpoint(
      filepath='saved_models/weights.best.{}.hdf5'.format(self.base_model_name),
      verbose=1, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001,
                                  patience=20, verbose=1)

    self.model.fit_generator(
      train_generator,
      steps_per_epoch=len(self.y_train) // batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=len(self.y_valid) // batch_size,
      callbacks=[checkpointer, early_stopping])

    # Load the model weights with the best validation loss.
    self.model.load_weights(
      'saved_models/weights.best.{}.hdf5'.format(self.base_model_name))


    self.bf_model = \
      Model(input=self.model.input, output=self.model.layers[:-1].output)

    self._save_model(self.model)

  def _extract_features(self, tensor):
    predicted_list = []
    for x in tensor:
      predicted_list.append(self.bf_model.predict(self.preprocess_input(x)))
    return np.array(predicted_list)

  def predict(self, tensor):
    predicted_list = []
    for x in tensor:
      predicted_list.append(self.model.predict(self.preprocess_input(x)))
    return np.array(predicted_list)

  def save_bottleneck_features(self,
                               inputs,
                               file_path,
                               batch_size=None,
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
          inputs_batch = utils.imgs_scale_to_255(inputs_batch).astype(
            data_type)

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
          bf_batch = self._extract_features(inputs_batch)
          f.write(pickle.dumps(bf_batch))

          # Release memory
          K.clear_session()
          tf.reset_default_graph()

      else:
        inputs = utils.imgs_scale_to_255(inputs).astype(data_type)

        if inputs_shape[3] == 1:
          inputs = np.concatenate([inputs, inputs, inputs], axis=-1)

        assert inputs.shape[1:] == (224, 224, 3)
        bottleneck_features = self._extract_features(inputs)
        f.write(pickle.dumps(bottleneck_features))


if __name__ == '__main__':

  FT = FineTune(
      config,
      n_output=148,
      base_model_name='xception', # 132层
      n_use_layers=None,
      n_freeze_layers=None,
      load_pre_model=False
  )
  FT.train(epochs=100, batch_size=16)
