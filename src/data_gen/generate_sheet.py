from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from os import path
from PIL import Image


class GenerateSheet(object):

  def __init__(self):

    self.source_data_dir = '../data/radicals/source'
    self.sheet_dir = '../data/sheet_canvas.jpg'
    self.save_img_dir = '../data/sheet'
    self.canvas_size = (1240, 1754)
    self.row_num = 10
    self.col_num = 15
    self.img_size = 70
    self.gap = 2

    self.canvas_left = int(
      (self.canvas_size[0] - (self.img_size + self.gap)
       * self.col_num - self.gap) // 2)
    self.canvas_top = int(
      self.canvas_size[1] - (self.img_size + self.gap)
      * self.row_num * 2 - self.canvas_left + 33)

    if not os.path.isdir(self.save_img_dir):
      os.makedirs(self.save_img_dir)

  def _select_image(self, class_name):

    class_dir = path.join(self.source_data_dir, class_name)
    img_names = os.listdir(class_dir)
    img_dir = path.join(
      class_dir, img_names[np.random.randint(0, len(img_names))])
    img = Image.open(img_dir).convert('L')
    img_width, img_height = img.size
    selected_image = Image.new('L', (self.img_size, self.img_size), 'white')
    if img_width > img_height:
      w_s = self.img_size
      h_s = int(w_s * img_height // img_width)
      img = img.resize((w_s, h_s), Image.ANTIALIAS)
      selected_image.paste(img, (0, int((self.img_size - h_s) // 2)))
    else:
      h_s = self.img_size
      w_s = int(h_s * img_width // img_height)
      img = img.resize((w_s, h_s), Image.ANTIALIAS)
      selected_image.paste(img, (int((self.img_size - w_s) // 2), 0))
    assert selected_image.size == (self.img_size, self.img_size)
    return selected_image

  def save_images(self, idx=0):

    canvas = Image.open(self.sheet_dir).convert('L')
    images = []
    count = 0

    class_names = [
      int(c) for c in os.listdir(self.source_data_dir) if c != '.DS_Store']
    class_names.sort()
    for class_name in class_names:
      images.append(self._select_image(str(class_name)))
    assert len(images) == 148

    horizontal_gap = Image.fromarray(
      np.zeros((self.gap, self.img_size + self.gap)), 'L')
    vertical_gap = Image.fromarray(
      np.zeros((self.img_size + self.gap, self.gap)), 'L')

    for row_i in range(self.row_num * 2):
      for col_i in range(self.col_num):
        if row_i % 2 == 0:
          if count < 148:
            img = images[count]
            canvas.paste(img, (
              int(col_i * (self.img_size + self.gap)
                  + self.canvas_left + self.gap),
              int(row_i * self.img_size +
                  (row_i + 1) * self.gap + self.canvas_top)
            ))
            count += 1

        canvas.paste(horizontal_gap, (
          int(col_i * (self.img_size + self.gap)
              + self.canvas_left),
          int(row_i * self.img_size +
              (row_i + 1) * self.gap + self.canvas_top - self.gap)
        ))
        canvas.paste(vertical_gap, (
          int(col_i * (self.img_size + self.gap)
              + self.canvas_left),
          int(row_i * self.img_size +
              (row_i + 1) * self.gap + self.canvas_top - self.gap)
        ))
        if col_i == self.col_num - 1:
          canvas.paste(vertical_gap, (
            int(self.col_num * (self.img_size + self.gap)
                + self.canvas_left),
            int(row_i * self.img_size +
                (row_i + 1) * self.gap + self.canvas_top - self.gap)
          ))
        if row_i == self.row_num - 1:
          canvas.paste(horizontal_gap, (
            int(col_i * (self.img_size + self.gap)
                + self.canvas_left),
            int(self.row_num * 2 * self.img_size +
                self.row_num * 2 * self.gap + self.canvas_top - self.gap)
          ))

    canvas.save(path.join(self.save_img_dir, str(idx) + '.jpg'))


if __name__ == '__main__':

  for i in range(10):
    GS = GenerateSheet()
    GS.save_images(i)
