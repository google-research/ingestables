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

"""Trainer lib."""

import pickle
from typing import Literal, Tuple

from absl import logging
from etils import epath
from ingestables.torch.data import encoders
from ingestables.torch.data import pipeline as pipeline_lib
from ingestables.torch.model import sklearn_model
from ingestables.torch.train import metrics
import numpy as np
import torch


class Trainer:
  """Manages the training loop.

  Example:
    trainer = Trainer(...)
    trainer.run()
  """

  def __init__(
      self,
      workdir: str,
      model: sklearn_model.SklearnModel,
      pipeline: pipeline_lib.Pipeline,
      metrics_writer: metrics.MetricsWriter | None = None,
      train_tabllm: bool = False,
  ):
    """Initializes the Trainer.

    Args:
      workdir: The working directory.
      model: The sklearn model to train.
      pipeline: The data pipeline to use.
      metrics_writer: The metrics writer to use.
      train_tabllm: Whether training a TabLLM model.
    """
    if metrics_writer is None:
      raise ValueError("metrics_writer must be set.")
    self._workdir = epath.Path(workdir)
    self._model = model
    self._pipeline = pipeline
    self._metrics_writer = metrics_writer
    self._train_tabllm = train_tabllm

  def __repr__(self):
    return (
        f"Trainer(workdir={self._workdir}, model={self._model},"
        f" pipeline={self._pipeline},"
        f" metrics_writer={self._metrics_writer})"
    )

  def _get_train_data(
      self,
      dataset_key: str,
  ) -> encoders.PreprocessedInputs:
    """Returns the train data loader for the given dataset."""
    return self._pipeline.get_train_data(dataset_key)[0:]

  def _get_test_data(
      self,
      dataset_key: str,
  ) -> encoders.PreprocessedInputs:
    """Returns the test data loader for the given dataset."""
    return self._pipeline.get_test_data(dataset_key)[0:]

  # TODO(mononito): We do not use val datasets yet
  def _get_val_data(
      self,
      dataset_key: str,
  ) -> encoders.PreprocessedInputs:
    """Returns the val data loader for the given dataset."""
    return self._pipeline.get_val_data(dataset_key)[0:]

  def run(self) -> None:
    """Train, evaluate, and save metrics."""
    for dataset_key in self._pipeline.dataset_keys:
      logging.info("Training on dataset: %s", dataset_key)
      batch = self._get_train_data(dataset_key)
      task_type = self._pipeline.get_task_info(dataset_key).task_type

      x, y = self._extract_features(batch)
      y = y.astype(int) if task_type == "classification" else y  # pylint: disable=attribute-error
      self._model.fit(x, y)

    # Evaluate.
    for dataset_key in self._pipeline.dataset_keys:
      logging.info("Evaluating test dataset: %s", dataset_key)
      eval_batch = self._get_test_data(dataset_key)
      self._evaluate("test", dataset_key, eval_batch)

      logging.info("Evaluating validation dataset: %s", dataset_key)
      eval_batch = self._get_val_data(dataset_key)
      self._evaluate("test", dataset_key, eval_batch)

    # Save model on last step
    self.save_model(self._model, self._workdir)
    logging.info("Job finished")

  def save_model(self, model, output_path: epath.Path):
    """Save model."""
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "model.sklearn").open("wb") as f:
      pickle.dump(model, f)

  def _to_numpy(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

  def _extract_features_tabllm(
      self,
      batch: encoders.PreprocessedInputs,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts features from the batch for TabLLM models."""

    assert batch.encoded_string is not None
    encoded_string = self._to_numpy(batch.encoded_string)
    encoded_string = encoded_string.reshape(encoded_string.shape[0], -1)

    y = self._to_numpy(batch.encoded_targets)

    return encoded_string, y

  def _extract_features_basic(
      self,
      batch: encoders.PreprocessedInputs,
  ) -> Tuple[np.ndarray | None, np.ndarray]:
    """Extracts categorical and numeric features from the batch to train basic skearn models.

    Args:
      batch: The batch to extract features from.

    Returns:
      The features as a 2D array and targets.
    """
    cat = None
    num = None
    if batch.encoded_numeric is not None:
      num = self._to_numpy(batch.encoded_numeric)
      num = num.reshape(num.shape[0], -1)
    if batch.encoded_categorical is not None:
      cat = self._to_numpy(batch.encoded_categorical)
      cat = cat.reshape(cat.shape[0], -1)

    y = self._to_numpy(batch.encoded_targets)

    if cat is not None and num is not None:
      return np.concatenate([cat, num], axis=1), y
    elif cat is None and num is None:
      raise ValueError("Batch has no numeric or categorical features.")
    else:
      return cat if cat is not None else num, y

  def _extract_features(
      self,
      batch: encoders.PreprocessedInputs,
  ) -> Tuple[np.ndarray | None, np.ndarray]:
    """Extracts features from the batch."""
    if self._train_tabllm:
      return self._extract_features_tabllm(batch)
    else:
      return self._extract_features_basic(batch)

  def _evaluate(
      self,
      dataset_type: Literal["test", "val"],
      dataset_key: str,
      eval_batch: encoders.PreprocessedInputs,
  ):
    """Evaluates the model on the given dataset."""
    task_type = self._pipeline.get_task_info(dataset_key).task_type

    x, y_trues = self._extract_features(eval_batch)
    y_probs = self._model.predict_proba(x)
    y_trues = y_trues.astype(int) if task_type == "classification" else y_trues  # pylint: disable=attribute-error

    mtrcs = self._model.compute_metrics(y_true=y_trues, y_probs=y_probs)
    self._metrics_writer.write_model_metrics(
        0, dataset_type, dataset_key, mtrcs
    )
    logging.info(
        "Metrics for dataset %s and task %s: %s",
        dataset_key,
        task_type,
        mtrcs,
    )
