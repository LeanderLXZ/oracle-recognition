from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import gc
import numpy as np
from PIL import Image
from os.path import isdir, join
from tqdm import tqdm


def save_data_to_pkl(data, data_path):
  """
  Save data to pickle file.
  """
  with open(data_path, 'wb') as f:
    pickle.dump(data, f)


class GenerateDataSet(object):

  def __init__(self, raw_data_dir, save_data_dir, img_size=(128, 128)):

    self.raw_data_dir = raw_data_dir
    self.save_data_dir = save_data_dir
    self.img_size = img_size
    contents = os.listdir(raw_data_dir)
    self.classes = [c for c in contents if isdir(join(raw_data_dir, c))]
    if not isdir(save_data_dir):
      os.mkdir(save_data_dir)

  def _reshape_img(self, img):

    reshaped_image = Image.new('L', self.img_size, 'white')
    img_width, img_height = img.size

    if img_width > img_height:
      w_s = self.img_size[0]
      h_s = int(w_s * img_height // img_width)
      img = img.resize((w_s, h_s), Image.ANTIALIAS)
      reshaped_image.paste(img, (0, int((self.img_size[1] - h_s) // 2)))
    else:
      h_s = self.img_size[1]
      w_s = int(h_s * img_width // img_height)
      img = img.resize((w_s, h_s), Image.ANTIALIAS)
      reshaped_image.paste(img, (int((self.img_size[0] - w_s) // 2), 0))

    reshaped_image = np.array(reshaped_image, dtype=np.float32)
    reshaped_image = reshaped_image.reshape((*reshaped_image.shape, 1))
    assert reshaped_image.shape == (*self.img_size, 1)
    return reshaped_image

  def generate(self):

    for class_ in tqdm(self.classes, ncols=100, unit='class'):
      class_dir = join(self.raw_data_dir, class_)
      images = os.listdir(class_dir)
      images_tensors = []

      for img_name in images:

        # Load image
        img = Image.open(join(class_dir, img_name)).convert('L')

        # Reshape image
        reshaped_img = self._reshape_img(img)

        # Save image to array
        images_tensors.append(reshaped_img)

      save_data_to_pkl(
          np.array(images_tensors, dtype=np.float32),
          join(self.save_data_dir, class_ + '.p'))

      del images_tensors
      gc.collect()


if __name__ == '__main__':

  if not isdir('../../data/source_data'):
    os.mkdir('../../data/source_data')

  GDS = GenerateDataSet(
      raw_data_dir='../../data/raw_data/radicals/total',
      save_data_dir='../../data/source_data/radical',
      img_size=(128, 128)
  )
  GDS.generate()
