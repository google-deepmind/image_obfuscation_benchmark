# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for the dataset."""

import enum
from typing import Callable, MutableMapping, Optional, Sequence

import tensorflow as tf
import tensorflow_datasets as tfds

CLEAN = 'Clean'

TRAIN_OBFUSCATIONS = [
    CLEAN,
    'AdversarialPatches',
    'BackgroundBlurComposition',
    'ColorNoiseBlocks',
    'Halftoning',
    'HighContrastBorder',
    'IconOverlay',
    'ImageOverlay',
    'Interleave',
    'InvertLines',
    'LineShift',
    'PerspectiveTransform',
    'PhotoComposition',
    'RotateBlocks',
    'RotateImage',
    'StyleTransfer',
    'SwirlWarp',
    'TextOverlay',
    'Texturize',
    'WavyColorWarp',
]

HOLD_OUT_OBFUSCATIONS = [
    'ColorPatternOverlay',
    'LowContrastTriangles',
    'PerspectiveComposition',
]

EVAL_OBFUSCATIONS = TRAIN_OBFUSCATIONS + HOLD_OUT_OBFUSCATIONS

_DATASET_NAME = 'obfuscated_imagenet'

_IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
_IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


# Type definitions.
_BATCH = MutableMapping[str, tf.Tensor]
_EVAL_SPLIT = 'validation'


class Normalization(enum.Enum):
  """What values the images are normalized to."""
  ZERO_ONE = 'zero_one'
  MINUS_PLUS_ONE = 'minus_plus_one'
  IMAGENET_CHANNEL_WISE_NORM = 'imagenet_channel_wise_norm'


class Split(enum.Enum):
  """Imagenet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @staticmethod
  def from_string(name: str) -> 'Split':
    return {
        'TRAIN': Split.TRAIN,
        'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
        'VALID': Split.VALID,
        'VALIDATION': Split.VALID,
        'TEST': Split.TEST
    }[name.upper()]


def get_obfuscations(split: Split) -> Sequence[str]:
  return EVAL_OBFUSCATIONS if split == Split.TEST else TRAIN_OBFUSCATIONS


def _uint8_to_unit_float(batch: _BATCH) -> _BATCH:
  batch['image'] = tf.cast(batch['image'], tf.float32) / 255.0
  return batch


def _uint8_to_neg1_pos1_float(batch: _BATCH) -> _BATCH:
  batch['image'] = 2 * tf.cast(batch['image'], tf.float32) / 255.0 - 1
  return batch


def _imagenet_channel_wise_norm(batch: _BATCH) -> _BATCH:
  batch['image'] = (tf.cast(batch['image'], tf.float32) -
                    _IMAGENET_MEAN) / _IMAGENET_STD
  return batch


def _get_normalize_fn(
    normalization: Normalization) -> Callable[[_BATCH], _BATCH]:
  if normalization == Normalization.ZERO_ONE:
    return _uint8_to_unit_float
  elif normalization == Normalization.MINUS_PLUS_ONE:
    return _uint8_to_neg1_pos1_float
  elif normalization == Normalization.IMAGENET_CHANNEL_WISE_NORM:
    return _imagenet_channel_wise_norm
  else:
    raise ValueError(f'Unknown normalization: `{normalization}`.')


def get_data(dataset_path: str,
             obfuscation: str,
             normalization: Normalization,
             batch_size: int = 32,
             dataset_version: Optional[str] = None,
             num_samples: int = 0) -> tf.data.Dataset:
  """Builds and returns the dataset."""
  if dataset_version:
    dataset_name = f'{_DATASET_NAME}:{dataset_version}'
  else:
    dataset_name = _DATASET_NAME
  ds = tfds.load(dataset_name,
                 data_dir=dataset_path,
                 split=f'{_EVAL_SPLIT}_{obfuscation}')
  normalize_fn = _get_normalize_fn(normalization)
  ds = ds.map(normalize_fn)
  if num_samples:
    ds = ds.take(num_samples)
  ds = ds.batch(batch_size)
  return ds
