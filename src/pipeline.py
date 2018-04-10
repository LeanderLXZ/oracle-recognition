from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import utils
from main import Main
from models.capsNet import CapsNet
from config_pipeline import *
from main import MainDistribute
from models.capsNet_distribute import CapsNetDistribute


def training_capsnet(cfg, mode):

  if mode == 'normal':
    CapsNet_ = CapsNet(cfg)
    Main_ = Main(CapsNet_, cfg)
    Main_.train()
  elif mode == 'multi-gpu':
    CapsNet_ = CapsNetDistribute(cfg)
    Main_ = MainDistribute(CapsNet_, cfg)
    Main_.train()
  else:
    raise ValueError('Wrong mode!')


def pipeline(mode):

  training_capsnet(cfg_1, mode)
  training_capsnet(cfg_2, mode)
  training_capsnet(cfg_3, mode)
  training_capsnet(cfg_4, mode)
  training_capsnet(cfg_5, mode)
  training_capsnet(cfg_6, mode)
  training_capsnet(cfg_7, mode)


if __name__ == '__main__':

  utils.thick_line()
  print('Input [ 1 ] to run normal version.')
  print('Input [ 2 ] to run multi-gpu version.')
  utils.thin_line()
  input_ = input('Input: ')

  if input_ == '1':
    pipeline('normal')
  elif input_ == '2':
    pipeline('multi-gpu')
  else:
    raise ValueError('Wrong input! Found: {}'.format(input_))
