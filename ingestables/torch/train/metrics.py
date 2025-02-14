# Copyright 2024 The ingestables Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics writer."""

import collections
from typing import Protocol

from absl import logging
from ingestables.torch import types
import numpy as np
import tensorflow as tf


class ComputeMetrics(Protocol):
  """Not really a head, but a way to compute metrics."""

  def compute_metrics(
      self,
      *,
      y_true: np.ndarray,
      y_probs: np.ndarray,
  ) -> types.Metrics:
    ...


class MetricsStore(Protocol):

  def write_metrics(
      self,
      step: int,
      key: tuple[str, ...],
      metrics: dict[str, float],
  ):
    ...


class TensorboardStore(MetricsStore):
  """Writes tensorboard summary."""

  def __init__(self, log_dir: str):
    self._log_dir = log_dir
    self._summary_writer = tf.summary.create_file_writer(log_dir)

  def __repr__(self):
    return f"TensorboardStore(log_dir={self._log_dir})"

  def write_metrics(
      self, step: int, key: tuple[str, ...], metrics: dict[str, float]
  ):
    with self._summary_writer.as_default():
      for name, value in metrics.items():
        tf.summary.scalar(f"{'/'.join(key)}/{name}", value, step=step)
      self._summary_writer.flush()


class InMemoryMetricsStore(MetricsStore):
  """Writes tensorboard summary."""

  def __init__(self):
    self._metrics = collections.defaultdict(dict)

  def __repr__(self):
    return "InMemoryMetricsStore()"

  @property
  def metrics(self) -> dict[int, dict[tuple[str, ...], dict[str, float]]]:
    """Step, dataset, head, metrics."""
    return self._metrics

  def metric(self, step: int, key: tuple[str, ...], metric_name: str) -> float:
    """Step, dataset, head, metrics."""
    return self._metrics[step][key][metric_name]

  def write_metrics(
      self,
      step: int,
      key: tuple[str, ...],
      metrics: dict[str, float],
  ):
    """Writes tensorboard summary."""
    step = self._metrics[step]
    if key not in step:
      step[key] = metrics
    else:
      step[key].update(metrics)


class MetricsWriter:
  """Writes tensorboard summary."""

  def __init__(self, store: MetricsStore):
    self._store = store

  def __repr__(self):
    return f"MetricsWriter(store={self._store})"

  def write_metrics(
      self,
      step: int,
      key: tuple[str, ...],
      metrics: dict[str, float],
  ):
    """Writes tensorboard summary."""
    logging.info("Writing step: %s, key: %s, metrics: %s", step, key, metrics)
    self._store.write_metrics(step, key, metrics)

  def write_metric(
      self,
      step: int,
      key: tuple[str, ...],
      name: str,
      value: float,
  ):
    self.write_metrics(step, key, {name: value})

  def write_model_metrics(
      self,
      step: int,
      dataset_type: str,
      dataset_key: str,
      metrics_type: types.Metrics,
  ):
    """Writes model metrics."""
    metrics = {
        f"{dataset_type}_{k}": v
        for k, v in metrics_type.metrics_dict().items()
        if v
    }
    logging.info(
        "Writing step: %s, dataset_type: %s, dataset_key: %s, head_key: %s,"
        " metrics: %s",
        step,
        dataset_type,
        dataset_key,
        metrics_type.head_key,
        metrics,
    )
    self.write_metrics(step, (dataset_key, metrics_type.head_key), metrics)
