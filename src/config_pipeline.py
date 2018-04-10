from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import copy
from easydict import EasyDict


# Auto-generate version
def _auto_version(c):
  _version = c['DATABASE_NAME']
  if c['WITH_RECONSTRUCTION']:
    _version += '_{}_{}'.format(c['DECODER_TYPE'], c['RECONSTRUCTION_LOSS'])
  else:
    _version += '_no_rec'
  if c['DPP_TEST_AS_VALID']:
    _version += '_tav'
  return _version


__C = EasyDict()

# ===========================================
# #             Hyperparameters             #
# ===========================================

# Database name
# 'mnist': MNIST
# 'cifar10' CIFAR-10
__C.DATABASE_NAME = 'cifar10'

# Training version
# Set None to auto generate version
__C.VERSION = None

# Learning rate
__C.LEARNING_RATE = 0.001

# Epochs
__C.EPOCHS = 50

# Batch size
__C.BATCH_SIZE = 256

# Setting test set as validation when preprocessing data
__C.DPP_TEST_AS_VALID = False

# ===========================================
# #            Model Architecture           #
# ===========================================

# -------------------------------------------
# Classification

# Parameters of margin loss
# default: {'m_plus': 0.9, 'm_minus': 0.1, 'lambda_': 0.5}
__C.MARGIN_LOSS_PARAMS = {'m_plus': 0.9,
                          'm_minus': 0.1,
                          'lambda_': 0.5}

# Add epsilon(a very small number) to zeros
__C.EPSILON = 1e-9

# stddev of tf.truncated_normal_initializer()
__C.WEIGHTS_STDDEV = 0.01

# -------------------------------------------
# Optimizer and learning rate decay

# Optimizer
# 'gd': GradientDescentOptimizer()
# 'adam': AdamOptimizer()
# 'momentum': MomentumOptimizer()
__C.OPTIMIZER = 'adam'

# Boundaries of learning rate
__C.LR_BOUNDARIES = [82, 123, 300]

# Stage of learning rate
__C.LR_STAGE = [1, 0.1, 0.01, 0.002]

# Momentum parameter of momentum optimizer
__C.MOMENTUM = 0.9

# -------------------------------------------
# Reconstruction

# Training with reconstruction
__C.WITH_RECONSTRUCTION = True

# Type of decoder of reconstruction:
# 'fc': full_connected layers
# 'conv': convolution layers
# 'conv_t': transpose convolution layers
__C.DECODER_TYPE = 'fc'

# Reconstruction loss
# 'mse': Mean Square Error
# 'ce' : sigmoid_cross_entropy_with_logits
__C.RECONSTRUCTION_LOSS = 'mse'

# Scaling for reconstruction loss
__C.RECONSTRUCT_LOSS_SCALE = 0.392  # 0.0005*784=0.392

# -------------------------------------------
# Test

# Evaluate on test set after training
__C.TEST_AFTER_TRAINING = True

# ===========================================
# #             Training Config             #
# ===========================================

# Display step
# Set None to not display
__C.DISPLAY_STEP = None  # batches

# Save summary step
# Set None to not save summaries
__C.SAVE_LOG_STEP = 20  # batches

# Save reconstructed images
# Set None to not save images
__C.SAVE_IMAGE_STEP = 50  # batches

# Maximum images number in a col
__C.MAX_IMAGE_IN_COL = 10

# Calculate train loss and valid loss using full data set
# 'per_epoch': evaluate on full set when n epochs finished
# 'per_batch': evaluate on full set when n batches finished
__C.FULL_SET_EVAL_MODE = 'per_batch'
# None: not evaluate
__C.FULL_SET_EVAL_STEP = 50

# Save models
# 'per_epoch': save models when n epochs finished
# 'per_batch': save models when n batches finished
__C.SAVE_MODEL_MODE = 'per_epoch'
# None: not save models
__C.SAVE_MODEL_STEP = 10
# Maximum number of recent checkpoints to keep.
__C.MAX_TO_KEEP_CKP = 5

# Calculate the train loss of full data set, which may take lots of time.
__C.EVAL_WITH_FULL_TRAIN_SET = False

# Show details of training progress
__C.SHOW_TRAINING_DETAILS = False

# ===========================================
# #             Testing Config              #
# ===========================================

# Testing version name
__C.TEST_VERSION = 'fc_rec_mse'

# Testing checkpoint index
__C.TEST_CKP_IDX = 29

# Testing with reconstruction
__C.TEST_WITH_RECONSTRUCTION = True

# Saving testing reconstruction images
# None: not save images
__C.TEST_SAVE_IMAGE_STEP = 10  # batches

# Batch size of testing
# should be same as training batch_size
__C.TEST_BATCH_SIZE = 256

# ===========================================
# #                  Others                 #
# ===========================================

if __C.VERSION is None:
  __C.VERSION = _auto_version(__C)

# Source data directory path
__C.SOURCE_DATA_PATH = '../data/source_data'

# Preprocessed data path
__C.DPP_DATA_PATH = '../data/preprocessed_data'

# Path for saving logs
__C.TRAIN_LOG_PATH = '../train_logs'

# Path for saving summaries
__C.SUMMARY_PATH = '../tf_logs'

# Path for saving models
__C.CHECKPOINT_PATH = '../checkpoints'

# Path for saving testing logs
__C.TEST_LOG_PATH = '../test_logs'

# Save trainable variables on CPU
__C.VAR_ON_CPU = True

# ===========================================
# #          Multi-GPUs Config              #
# ===========================================

# Number of GPUs
__C.GPU_NUMBER = 2

# Batch size on a single GPU
__C.GPU_BATCH_SIZE = __C.BATCH_SIZE // __C.GPU_NUMBER

# The decay to use for the moving average.
__C.MOVING_AVERAGE_DECAY = 0.9999

# ===========================================

# get config by: from distribute_config import config
config = __C

# ===========================================
# #                 Pipeline                #
# ===========================================

__C.WITH_RECONSTRUCTION = False
__C.VERSION = _auto_version(__C)
cfg_1 = copy(__C)

__C.WITH_RECONSTRUCTION = True
__C.VERSION = _auto_version(__C)
cfg_2 = copy(__C)

__C.RECONSTRUCTION_LOSS = 'ce'
__C.VERSION = _auto_version(__C)
cfg_3 = copy(__C)

__C.DECODER_TYPE = 'conv'
__C.RECONSTRUCTION_LOSS = 'mse'
__C.VERSION = _auto_version(__C)
cfg_4 = copy(__C)

__C.RECONSTRUCTION_LOSS = 'ce'
__C.VERSION = _auto_version(__C)
cfg_5 = copy(__C)

__C.DECODER_TYPE = 'conv_t'
__C.RECONSTRUCTION_LOSS = 'mse'
__C.VERSION = _auto_version(__C)
cfg_6 = copy(__C)

__C.RECONSTRUCTION_LOSS = 'ce'
__C.VERSION = _auto_version(__C)
cfg_7 = copy(__C)
