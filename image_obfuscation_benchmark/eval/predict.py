#!/usr/bin/python
#
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


"""Write predictions from a model for the benchmark."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from image_obfuscation_benchmark.eval import data_utils
from image_obfuscation_benchmark.eval import evaluate_lib

_DATASET_PATH = flags.DEFINE_string(
    'dataset_path', None, 'Path to the dataset.', required=True)
_MODEL_PATH = flags.DEFINE_string(
    'model_path', None, 'Path to the exported model. Can be a TF Hub address.',
    required=True)
_EVALUATE_OBFUSCATION = flags.DEFINE_string(
    'evaluate_obfuscation', None, 'On what obfuscation to evaluate on.',
    required=True)
_NORMALIZATION = flags.DEFINE_enum_class(
    'normalization', data_utils.Normalization.ZERO_ONE,
    data_utils.Normalization,
    'How to normalize the images. Either `zero_one` ([0, 1]), `minus_plus_one` '
    '([-1, 1]) or `imagenet_channel_wise_norm` (using ImageNet mean and std.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', '/tmp/', 'Where to save the predictions.')
_USE_CLASS_GROUPS = flags.DEFINE_bool(
    'use_class_groups', True,
    'Whether to use the stylized imagenet / conflict stimuli class groups.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset_path = _DATASET_PATH.value
  model_path = _MODEL_PATH.value
  obfuscation = _EVALUATE_OBFUSCATION.value
  normalization = _NORMALIZATION.value

  logging.info('Loading dataset for obfuscation `%s` from `%s`.',
               obfuscation, dataset_path)
  ds = data_utils.get_data(dataset_path=dataset_path,
                           obfuscation=obfuscation,
                           normalization=normalization)

  logging.info('Loading model from `%s`.', model_path)
  model = evaluate_lib.load_model(model_path)
  image_ids, labels, predictions = evaluate_lib.predict(
      ds, model, _USE_CLASS_GROUPS.value)
  filename = os.path.join(_OUTPUT_DIR.value, f'{obfuscation}.csv')
  logging.info('Saving predictions to `%s`.', filename)
  evaluate_lib.save_predictions(image_ids, labels, predictions, filename)


if __name__ == '__main__':
  app.run(main)
