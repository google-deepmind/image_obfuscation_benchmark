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

"""Tests for the image obfuscation evaluation library."""


from absl.testing import absltest
from absl.testing import parameterized
from image_obfuscation_benchmark.eval import data_utils
from image_obfuscation_benchmark.eval import evaluate_lib
import mock
import numpy as np
import tensorflow as tf


class EvaluateLibTest(parameterized.TestCase):

  def setUp(self):
    super(EvaluateLibTest, self).setUp()
    images = [0, 1, 2, 3, 4]
    self._image_ids = [b'a', b'b', b'c', b'd', b'e']
    self._labels = [11, 440, 409, 294, 15]
    self._super_class_labels = [3, 5, 9, 1, 3]
    self._top_n_predictions = {0: [11, 12, 9, 555, 766],
                               1: [720, 440, 436, 281, 424],
                               2: [404, 409, 152, 385, 386],
                               3: [1, 2, 8, 152, 404],
                               4: [1, 2, 15, 385, 294]}
    self._dataset = tf.data.Dataset.from_tensor_slices(
        {'file_name': self._image_ids, 'label': self._labels, 'image': images}
    ).batch(1)

    def _logit_function(tensor):
      logits = np.zeros((1, 1000))
      for i, n in enumerate(self._top_n_predictions[tensor.numpy()[0]]):
        logits[0, n] = 100 - i * 10
      return tf.constant(logits, dtype=tf.float32)

    self._model = mock.MagicMock(side_effect=_logit_function)

  @parameterized.parameters([
      dict(use_class_groups=False, top_n=1,
           expected_predictions=[[11], [720], [404], [1], [1]]),
      dict(use_class_groups=False, top_n=2,
           expected_predictions=[[11, 12], [720, 440], [404, 409], [1, 2],
                                 [1, 2]]),
      dict(use_class_groups=False, top_n=3,
           expected_predictions=[[11, 12, 9], [720, 440, 436], [404, 409, 152],
                                 [1, 2, 8], [1, 2, 15]]),
      dict(use_class_groups=True, top_n=1,
           expected_predictions=[[3], [5], [0], [3], [3]]),
      dict(use_class_groups=True, top_n=2,
           expected_predictions=[[3, 15], [5, 6], [0, 9], [3, 10], [3, 11]]),
      dict(use_class_groups=True, top_n=3,
           expected_predictions=[[3, 15, 14], [5, 6, 7], [0, 9, 10], [3, 10, 0],
                                 [3, 11, 1]]),
  ])
  def test_predict(self, use_class_groups, top_n, expected_predictions):
    image_ids, labels, predictions = evaluate_lib.predict(
        self._dataset, self._model,
        use_class_groups=use_class_groups,
        top_n=top_n)

    np.testing.assert_equal(image_ids, self._image_ids)
    if use_class_groups:
      np.testing.assert_equal(labels, self._super_class_labels)
    else:
      np.testing.assert_equal(labels, self._labels)
    np.testing.assert_equal(predictions.shape,
                            (len(self._image_ids), top_n))
    np.testing.assert_equal(predictions, expected_predictions)

  def test_calculate_metrics(self):
    label_dict = {}
    predictions_dict = {}

    for obfuscation in data_utils.EVAL_OBFUSCATIONS:
      label_dict[obfuscation] = np.array(self._labels)
      predictions_dict[obfuscation] = np.array(
          list(self._top_n_predictions.values()))

    metrics = evaluate_lib.calculate_metrics(
        label_dict, predictions_dict, label_weighted=False, top_ns=(1, 2, 3))

    top_1 = 0.2
    top_2 = 0.6
    top_3 = 0.8
    expected_metrics = {}

    for prefix in ('mean-training', 'mean-hold-out', 'mean-all',
                   'worst-training', 'worst-hold-out', 'worst-all'):
      expected_metrics[f'{prefix}-top-1'] = top_1
      expected_metrics[f'{prefix}-top-2'] = top_2
      expected_metrics[f'{prefix}-top-3'] = top_3

    for obfuscation in data_utils.EVAL_OBFUSCATIONS:
      expected_metrics[f'{obfuscation}-top-1'] = top_1
      expected_metrics[f'{obfuscation}-top-2'] = top_2
      expected_metrics[f'{obfuscation}-top-3'] = top_3

    self.assertEqual(metrics, expected_metrics)

if __name__ == '__main__':
  absltest.main()
