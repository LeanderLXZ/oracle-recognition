import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

model_name = 'xception'
LAYERS_TO_FREEZE = 3

if model_name == 'vgg16':
  from keras.applications.vgg16 import VGG16, preprocess_input
elif model_name == 'vgg19':
  from keras.applications.vgg19 import VGG19, preprocess_input
elif model_name == 'resnet50':
  from keras.applications.resnet50 import ResNet50, preprocess_input
elif model_name == 'inceptionv3':
  from keras.applications.inception_v3 import InceptionV3, preprocess_input
elif model_name == 'xception':
  from keras.applications.xception import Xception, preprocess_input

base_model = InceptionV3(weights='imagenet', include_top=False)
for layer in base_model.layers[:LAYERS_TO_FREEZE]:
  layer.trainable = False
for layer in base_model.layers[LAYERS_TO_FREEZE:]:
  layer.trainable = True