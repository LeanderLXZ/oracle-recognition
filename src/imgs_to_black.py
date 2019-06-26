#!/usr/bin/env python
import os
from os.path import join
from models import utils
from tqdm import tqdm
from PIL import Image
import numpy as np

# source_data_path = '../data/source_data/radical'
# save_data_path = '../data/source_data/radical_black'

# classes = os.listdir(source_data_path)

# if '.DS_Store' in classes:
# 	classes.remove('.DS_Store')
# classes = sorted([int(i) for i in classes])

# for cls_ in tqdm(classes, ncols=100, unit='class'):
# 	cls_name = str(cls_)
# 	class_dir = join(source_data_path, cls_name)
# 	save_class_dir = join(save_data_path, cls_name)
# 	utils.check_dir([save_class_dir])

# 	images = os.listdir(class_dir)
# 	for img_name in images:
# 		img = Image.open(join(class_dir, img_name)).convert('L')
# 		img = 255 - np.array(img)
# 		img = Image.fromarray(img.astype('uint8'), 'L')
# 		img.save(join(save_class_dir, img_name))


# source_data_path = '../data/source_data/recognized_oracles'
# save_data_path = '../data/source_data/recognized_oracles_black'

# classes = os.listdir(source_data_path)

# utils.check_dir([save_data_path])

# images = os.listdir(source_data_path)
# for img_name in images:
# 	img = Image.open(join(source_data_path, img_name)).convert('L')
# 	img = 255 - np.array(img)
# 	img = Image.fromarray(img.astype('uint8'), 'L')
# 	img.save(join(save_data_path, img_name))

source_data_path = '../data/source_data/radical'
save_data_path = '../data/source_data/radical_1'
utils.check_dir([save_data_path])

classes = os.listdir(source_data_path)

if '.DS_Store' in classes:
	classes.remove('.DS_Store')
classes = sorted([int(i) for i in classes])

for cls_ in tqdm(classes, ncols=100, unit='class'):
	cls_name = str(cls_)
	class_dir = join(source_data_path, cls_name)

	images = os.listdir(class_dir)
	img_name = images[0]
	img = Image.open(join(class_dir, img_name)).convert('L')
	img = 255 - np.array(img)
	img = Image.fromarray(img.astype('uint8'), 'L')
	img.save(join(save_data_path, cls_name+'.jpg'))
