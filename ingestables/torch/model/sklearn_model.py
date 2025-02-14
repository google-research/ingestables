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

"""Holds sklearn model and related classes/functions.

In order to keep the codebase consistent, sklearn models will need other objects
to be consistent with the torch models. For example, the sklearn model will not
have a 'head' where metrics are computed, but we will have an analogous metrics
object that can be used to compute metrics.

Plus this will also allow us to have one interface for all the sklearn models
and handle the intricacies of each model type. This can also be a basis for
an interface that can switch between sklearn and torch models.
"""

from typing import Any

from ingestables.torch import types
from ingestables.torch.train import metrics
import numpy as np
import torch


class SklearnModel(metrics.ComputeMetrics):
  """Holds sklearn model and related classes/functions."""

  def __init__(self, wrapped_model: Any, head: metrics.ComputeMetrics):
    """Initialize the SklearnModel.

    Args:
      wrapped_model: The sklearn model.
      head: The head to compute loss and metrics.
    """
    self._wrapped_model = wrapped_model
    self._head = head

  def __repr__(self):
    return f"SklearnModel(model={self._wrapped_model}, head={self._head})"

  def wrapped_model(self) -> Any:
    return self._wrapped_model

  def compute_metrics(
      self,
      y_true: np.ndarray,
      y_probs: np.ndarray,
  ) -> types.Metrics:
    return self._head.compute_metrics(y_true=y_true, y_probs=y_probs)

  def fit(self, x: np.ndarray, y: np.ndarray):
    # TODO(joetoth): Conditional hack as we ensure the input to this is all
    # numpy or all torch.
    if isinstance(x, torch.Tensor):
      x = x.numpy()
    self._wrapped_model.fit(x, y)

  def predict_proba(self, x):
    """Return prediction probabilities for each class of each output."""
    if hasattr(self._wrapped_model, "predict_proba"):
      return self._wrapped_model.predict_proba(x)
    if hasattr(self._wrapped_model, "predict"):
      return self._wrapped_model.predict(x)
    raise ValueError("Model does not have predict_proba or predict")
