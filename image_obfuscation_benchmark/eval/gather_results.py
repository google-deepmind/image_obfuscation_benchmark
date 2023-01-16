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

"""Gathers the results for all obfuscations and calculates metrics."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from image_obfuscation_benchmark.eval import evaluate_lib
import tensorflow as tf

_LABEL_WEIGHTED = flags.DEFINE_bool(
    'label_weighted', True,
    'Whether accuracy should be weighted equally across the labels.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Where to save the predictions.', required=True)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  label_dict, predictions_dict = evaluate_lib.gather_predictions(
      _OUTPUT_DIR.value)

  metrics = evaluate_lib.calculate_metrics(
      label_dict, predictions_dict, _LABEL_WEIGHTED.value)
  with tf.gfile.Open(os.path.join(_OUTPUT_DIR.value, 'metrics.csv'), 'w') as f:
    for name, value in metrics.items():
      logging.info('%s: %.2f%%', name, 100 * value)
      f.write(f'{name}, {value}\n')


if __name__ == '__main__':
  app.run(main)
