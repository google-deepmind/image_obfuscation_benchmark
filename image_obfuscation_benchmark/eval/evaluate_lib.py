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

"""Functions to evaluates a model for the benchmark."""

import functools
import os
from typing import Callable, Mapping, Sequence, Tuple

from absl import logging
from image_obfuscation_benchmark.eval import data_utils
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_hub as hub
import tqdm

_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_TOP_N_METRICS = (1, 3, 5, 10)
_MAX_TOP_N = max(_TOP_N_METRICS)


_CONFLICT_STIMULI_GROUPS = [
    [404],
    [294, 295, 296, 297],
    [444, 671],
    [8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 80, 81, 82, 83, 87,
     88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 127, 128, 129, 130, 131,
     132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145],
    [472, 554, 625, 814, 914],
    [440, 720, 737, 898, 899, 901, 907],
    [436, 511, 817],
    [281, 282, 283, 284, 285, 286],
    [423, 559, 765, 857],
    [409, 530, 892],
    [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
     167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
     182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197,
     198, 199, 200, 201, 202, 203, 205, 206, 207, 208, 209, 210, 211, 212, 213,
     214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 229,
     230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 245,
     246, 247, 248, 249, 250, 252, 253, 254, 255, 256, 257, 259, 261, 262, 263,
     265, 266, 267, 268],
    [385, 386],
    [508, 878],
    [499],
    [766],
    [555, 569, 656, 675, 717, 734, 864, 867],
]


def _load_hub_model(model_path: str) -> tf.keras.Model:
  model = tf.keras.Sequential([hub.KerasLayer(model_path)])
  model.build([None, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS])
  return model


def _arg_top_n(values, n):
  return np.flip(np.argsort(values)[:, -n:], axis=-1)


def _group_logits(
    logits: np.ndarray,
    aggregate_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
  """Groups logits together to use the 16 classes from Stylized ImageNet."""
  new_logits = np.zeros((logits.shape[0], len(_CONFLICT_STIMULI_GROUPS)))
  probabilities = scipy.special.softmax(logits, axis=-1)
  for i, class_group in enumerate(_CONFLICT_STIMULI_GROUPS):
    new_logits[:, i] = aggregate_fn(probabilities[:, class_group])
  return new_logits


def _get_predictions(images: tf.Tensor, model: tf.keras.Model, top_n: int = 10,
                     use_class_groups: bool = False) -> np.ndarray:
  # Some models return 1001 logits per example. The first one is for a
  # background class, the rest are the 1000 ImageNet classes.
  logits = model(images).numpy()[:, -1000:]
  if use_class_groups:
    logits = _group_logits(logits, functools.partial(np.mean, axis=-1))
  return _arg_top_n(logits, top_n)


def _filter_class_groups(
    image_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Filters by images that contain the class groups and adjusts."""
  all_classes = []
  for group in _CONFLICT_STIMULI_GROUPS:
    for im_class in group:
      all_classes.append(im_class)
  all_classes = np.array(all_classes)
  filter_ids = np.isin(labels, all_classes)
  logging.info('Filtering images not in class groups:')
  logging.info('Before: %d, after: %d.', len(filter_ids), np.sum(filter_ids))
  labels = labels[filter_ids]
  image_ids = image_ids[filter_ids]
  predictions = predictions[filter_ids]
  new_labels = -1 * np.ones_like(labels)
  for i, class_group in enumerate(_CONFLICT_STIMULI_GROUPS):
    new_labels[np.isin(labels, class_group)] = i
  return image_ids, new_labels, predictions


def load_model(model_path: str) -> tf.keras.Model:
  if model_path.startswith('http') or model_path.startswith('@'):
    model = _load_hub_model(model_path)
  else:  # We assume it's a filepath.
    model = tf.saved_model.load(model_path)
  return model


def predict(dataset: tf.data.Dataset,
            model: tf.keras.Model,
            use_class_groups: bool = False,
            top_n: int = _MAX_TOP_N
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Produces predictions for a specific dataset ordered by image_ids."""
  image_ids = []
  labels = []
  predictions = []
  for batch in tqdm.tqdm(dataset, desc='Making predictions'):
    image_ids.extend(batch['original_id'].numpy().tolist())
    labels.extend(list(batch['label']))
    predictions.extend(
        list(_get_predictions(batch['image'],
                              model,
                              top_n=top_n,
                              use_class_groups=use_class_groups)))
  sort_idx = np.argsort(image_ids)
  image_ids = np.array(image_ids)[sort_idx]
  labels = np.array(labels)[sort_idx]
  predictions = np.array(predictions)[sort_idx]
  if use_class_groups:
    image_ids, labels, predictions = _filter_class_groups(
        image_ids, labels, predictions)
  return image_ids, labels, predictions


def save_predictions(image_ids: np.ndarray,
                     labels: np.ndarray,
                     predictions: np.ndarray,
                     filename: str):
  if not tf.gfile.Exists(os.path.dirname(filename)):
    tf.gfile.MakeDirs(os.path.dirname(filename))
  with tf.gfile.Open(filename, 'w') as f:
    for image_id, label, prediction in zip(image_ids, labels, predictions):
      f.write(f'{image_id}, {label}, {prediction}\n')


def load_predictions(
    filename: str,
    dtype: np.dtype = np.int32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Loads predictions from a csv file."""
  image_ids = []
  labels = []
  predictions = []
  with tf.gfile.Open(filename, 'r') as f:
    for line in f:
      line = line.rstrip('\n')
      if line:
        values = line.split(',')
        image_ids.append(values[0])
        labels.append(int(values[1]))
        predictions.append(
            np.fromstring(values[2].strip(' []'), dtype=dtype, sep=' '))
  return np.array(image_ids), np.array(labels), np.array(predictions)


def gather_predictions(
    filedir: str,
    dtype: np.dtype = np.int32) -> Tuple[Mapping[str, np.ndarray],
                                         Mapping[str, np.ndarray]]:
  """Loads predictions for all obfuscations."""

  image_id_dict = {}
  label_dict = {}
  predictions_dict = {}
  for obfuscation in data_utils.get_obfuscations(data_utils.Split.TEST):
    filename = os.path.join(filedir, f'{obfuscation}.csv')
    logging.info('Loading predictions for obfuscation `%s` from `%s`.',
                 obfuscation, filename)
    image_ids, labels, predictions = load_predictions(filename, dtype)
    image_id_dict[obfuscation] = image_ids
    label_dict[obfuscation] = labels
    predictions_dict[obfuscation] = predictions
  # Predictions should be sorted by image_ids so they should all match up.
  assert data_utils.CLEAN in image_id_dict
  for image_ids in image_id_dict.values():
    assert (image_id_dict[data_utils.CLEAN] == image_ids).all()
  return label_dict, predictions_dict


def _is_in_top_n(
    labels: np.ndarray, predictions: np.ndarray, n: int) -> np.ndarray:
  return (labels[..., None] == predictions[:, :n]).any(1)


def _calculate_mean(correctness: np.ndarray, labels: np.ndarray,
                    label_weighted: bool = False) -> float:
  if not label_weighted:
    return np.mean(correctness)
  correctness = correctness.flatten()
  labels = labels.flatten()
  label_counts = np.bincount(labels)
  weights = [1.0 / float(label_counts[label]) for label in labels]
  return float(np.average(correctness, weights=weights))


def calculate_metrics(
    label_dict: Mapping[str, np.ndarray],
    predictions_dict: Mapping[str, np.ndarray],
    label_weighted: bool = False,
    top_ns: Sequence[int] = _TOP_N_METRICS) -> Mapping[str, float]:
  """Calculates the metrics from predictions and labels."""
  metrics = {}
  num_hold_out = len(data_utils.HOLD_OUT_OBFUSCATIONS)

  for n in top_ns:
    correctness_table = []
    label_table = []
    for obfuscation in data_utils.get_obfuscations(data_utils.Split.TEST):
      correctness_table.append(_is_in_top_n(
          label_dict[obfuscation], predictions_dict[obfuscation], n))
      label_table.append(label_dict[obfuscation])
    correctness_table = np.stack(correctness_table, axis=0)
    label_table = np.stack(label_table, axis=0)

    for i, obfuscation in enumerate(label_dict):
      metrics[f'{obfuscation}-top-{n}'] = _calculate_mean(
          correctness_table[i, :], label_table[i, :], label_weighted)

    metrics[f'mean-training-top-{n}'] = _calculate_mean(
        correctness_table[1:-num_hold_out, :],
        label_table[1:-num_hold_out, :], label_weighted)
    metrics[f'mean-hold-out-top-{n}'] = _calculate_mean(
        correctness_table[-num_hold_out:, :],
        label_table[-num_hold_out:, :], label_weighted)
    metrics[f'mean-all-top-{n}'] = _calculate_mean(
        correctness_table[1:, :], label_table[1:, :], label_weighted)
    metrics[f'worst-training-top-{n}'] = _calculate_mean(
        np.all(correctness_table[1:-num_hold_out, :], axis=0),
        label_table[0, :], label_weighted)
    metrics[f'worst-hold-out-top-{n}'] = _calculate_mean(
        np.all(correctness_table[-num_hold_out:, :], axis=0),
        label_table[0, :], label_weighted)
    metrics[f'worst-all-top-{n}'] = _calculate_mean(
        np.all(correctness_table[1:, :], axis=0),
        label_table[0, :], label_weighted)
  return metrics
